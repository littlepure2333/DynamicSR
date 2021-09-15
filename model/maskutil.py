import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask:
    """hard and soft mask [default=None]"""

    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape
        self.hard = hard
        self.soft = soft
        self.active_positions = torch.sum(hard)  # active positions
        self.total_positions = hard.numel()  # total positions
        self.flops_per_position = 0  # flops per position

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, ' \
               f'and {self.flops_per_position} accumulated FLOPS per position '


class MaskUnit(nn.Module):
    """Generates the mask and applies the gumbel softmax trick"""

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(MaskUnit, self).__init__()
        # self.maskconv = Squeeze(channels=channels, stride=stride)  # squeeze
        self.maskconv = nn.Conv2d(channels, 1, stride=stride, kernel_size=1, padding=0)
        self.gumbel = Gumbel()  # gumbel
        self.expandmask = ExpandMask(stride=dilate_stride)  # expand

    def forward(self, x, meta):
        soft = self.maskconv(x)  # ->soft
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])  # soft->hard
        mask = Mask(hard, soft)  # (hard,soft)->mask
        hard_dilate = self.expandmask(mask.hard)  # hard dilate
        mask_dilate = Mask(hard_dilate)  # (hard dilate,None)->mask dilate
        m = {'std': mask, 'dilate': mask_dilate}  # mask, mask dilate
        meta['masks'].append(m)
        return m


class Gumbel(nn.Module):
    """Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x."""

    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:
            return (x >= 0).float()
        # logger.add('gumbel_noise', gumbel_noise)
        # logger.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps) + eps), - \
                torch.log(-torch.log(U2 + eps) + eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard


class Squeeze(nn.Module):
    """ Squeeze module to predict masks """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (b,c,h,w)->(b,c)
        self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)  # (b,1,1,1)
        z = self.conv(x)  # (b,c,h,w)->(b,1,h,w)
        return z + y.expand_as(z)  # (b,1,h,w)+(b,1,1,1)


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1):
        super(ExpandMask, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros((1, 1, self.stride, self.stride), device=x.device)
            self.pad_kernel[0, 0, 0, 0] = 1
        self.dilate_kernel = torch.ones((1, 1, 1 + 2 * self.padding, 1 + 2 * self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5


def conv1x1_mask(conv_module, x, mask, fast=False):
    """mask conv1x1, need gathering and splatting"""
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0] * w.shape[1]  # flops per position
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3_dw_mask(conv_module, x, mask_dilate, mask, fast=False):
    """mask conv3x3 depth-wise, need gathering and splatting"""
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3_mask(conv_module, x, mask_dilate, mask, fast=False):
    """mask conv3x3, need gathering and splatting"""
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


def bn_relu_mask(bn_module, relu_module, x, mask, fast=False):
    """bn+relu, need gathering and splatting"""
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask

    x = bn_module(x)
    x = relu_module(x) if relu_module is not None else x
    return x


def apply_mask(x, mask):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x  # apply mask


def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        m = mask['std'].hard
        assert m.dim() == 4
        m = m[0]  # only show the first image of the batch
        if out is None:
            out = m
        else:
            out += F.interpolate(m.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ResBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        # conv+relu+conv
        self.conv1 = conv3x3(n_feats, n_feats, stride=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(n_feats, n_feats, stride=1, bias=bias)
        self.res_scale = res_scale
        self.masker = MaskUnit(channels=n_feats, stride=1, dilate_stride=1)

    def forward(self, input):
        x, meta = input
        assert meta is not None
        identity = x  # identity
        m = self.masker(x, meta)  # mask and mask dilate
        mask_dilate, mask = m['dilate'], m['std']
        out = conv3x3_mask(self.conv1, x, None, mask_dilate)  # conv1
        out = self.act(out)  # act1
        out = conv3x3_mask(self.conv2, out, mask_dilate, mask)  # conv2
        res = out.mul(self.res_scale)
        res = apply_mask(res, mask)  # res*mask
        res += identity
        return res, meta



