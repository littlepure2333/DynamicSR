import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from model import common

"""Enhanced Deep Residual Networks for Single Image Super-Resolution"""


def make_model(args, parent=False):
    return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.n_resblocks = n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.ada_depth = args.ada_depth

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlockSA(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        self.last_conv = conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.predictor = Predictor(channel_in=args.n_colors, n_feats=n_feats, n_layers=5, reduction=4, nc_adapter=1,
                                   upper_bound=n_resblocks)  # depth predictor

    def forward(self, x):
        b, _, h, w = x.shape
        
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        if self.training:
            depth = torch.empty([b, 1], device=x.device).uniform_(0, self.n_resblocks)
        else:
            depth = torch.empty([b, 1], device=x.device).fill_(self.ada_depth)

        # predict depth-map: [B, 1, H, W]
        pred = self.predictor(x, depth)

        # body
        for i, layer in enumerate(self.body):
            mask_i = torch.clamp(pred - i, 0, 1)
            res = layer(res, mask_i)
        res = self.last_conv(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, pred, depth

class Predictor(nn.Module):
    """depth predictor"""

    def __init__(self, channel_in, n_feats, n_layers=5,
                 reduction=4, nc_adapter=1, upper_bound=float('inf')):
        super(Predictor, self).__init__()
        self.n_feats, self.n_layers = n_feats, n_layers
        self.upper_bound = upper_bound
        pred_feats = n_feats // reduction

        layers = [nn.Conv2d(n_feats, pred_feats, 3, stride=1, padding=1)]  # c_ft->c_red
        for _ in range(n_layers - 2):
            layers += [nn.PReLU(), nn.Conv2d(pred_feats, pred_feats, 3, stride=1, padding=1)]  # (c_red->c_red)x(N-2)
        layers += [nn.PReLU(), nn.Conv2d(pred_feats, nc_adapter, 3, stride=1, padding=1)]  # c_red->c_ada
        self.conv = nn.Sequential(*layers)

    def forward(self, x, depth=None):
        x = self.conv(x)
        if depth is not None:
            x = x * depth.view(-1, 1, 1, 1)  # (b,1,h,w)
        return x.clamp(0, self.upper_bound)  # clamp to (0,max)


class BasicBlockSA(nn.Module):
    """conv+norm+relu+conv+norm"""

    def __init__(self, dim, norm_type, clamp=None, attention='ca', use_depth=False):
        super().__init__()
        self.clamp, self.attention, self.use_depth = clamp, attention, use_depth
        norm_layer = get_norm_layer(norm_type)

        # blocks = [nn.ReLU(inplace=True), nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
        blocks = [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
        blocks += [nn.ReLU(inplace=True), nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
        self.ft = nn.Sequential(*blocks)
        self.ft.apply(init_weights)

        self.ca = CALayer(channel=dim, reduction=4)
        # self.sa = SALayer()

    def forward(self, x, depth=None):
        if depth is None:
            d_map = torch.ones((1, 1, *x.shape[-2:]), device=x.device)
        else:
            d_map = self.clamp(depth)

        res = self.ft(x)
        res = res * d_map if self.use_depth else res
        # res = self.ca(res)  # attention
        return res + x


if __name__ == "__main__":
    print(__file__)
