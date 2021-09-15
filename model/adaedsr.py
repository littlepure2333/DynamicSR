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
        n_resblocks = args.n_resblocks
        k, n_feats, scale = 3, args.n_feats, int(args.scale[0])
        act = nn.ReLU(True)
        self.n_resblocks = n_resblocks

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head
        m_head = [conv(args.n_colors, n_feats, k)]  # 3->n_ft

        # body, (n_ft->n_ft) x n_resblocks
        self.body = nn.ModuleList()
        for i in range(n_resblocks):
            self.body.append(ResBlock(conv, n_feats, k, clamp=clamp_wrapper(i), act=act, res_scale=args.res_scale))
        self.body_last = conv(n_feats, n_feats, k)

        # tail
        m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, 3, k)]

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

        self.predictor = Predictor(channel_in=args.n_colors, n_feats=n_feats, n_layers=5, reduction=4, nc_adapter=1,
                                   upper_bound=n_resblocks)  # depth predictor
        self.ratio = 0

    def forward(self, x, krl=None, meta=None):
        # x, krl, meta
        # -mean
        x = self.sub_mean(x)
        # depth
        batch_size = x.shape[0]

        if self.training:
            depth = torch.empty([batch_size, 1], device=x.device).uniform_(0, self.n_resblocks)
        else:
            # depth = torch.empty([batch_size, 1], device=x.device).fill_(self.n_resblocks)
            depth = torch.empty([batch_size, 1], device=x.device).fill_(self.n_resblocks*0.9)

        pred = self.predictor(x, depth=depth)  # predict depth

        b, _, h, w = pred.shape
        self.ratio = pred.sum() / (self.n_resblocks * b * h * w)
        # head
        x = self.head(x)
        # body
        res = self.body[0](x, pred)
        for i in range(self.n_resblocks - 1):
            res = self.body[i + 1](res, pred)

        # res = self.body[0](x, None)
        # for i in range(self.n_resblocks - 1):
        #     res = self.body[i + 1](res, None)

        res = self.body_last(res)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)

        return x, pred, depth


def clamp_wrapper(i):
    """clamp wrapper, clamp x-i to [0,1]"""

    def clamp(x):
        return torch.clamp(x - i, 0, 1)

    return clamp


class Predictor(nn.Module):
    """depth predictor"""

    def __init__(self, channel_in, n_feats, n_layers=5,
                 reduction=4, nc_adapter=1, upper_bound=float('inf')):
        super(Predictor, self).__init__()
        self.n_feats, self.n_layers = n_feats, n_layers
        self.upper_bound = upper_bound
        pred_feats = n_feats // reduction
        layers = [nn.Conv2d(channel_in, n_feats, kernel_size=3, padding=1), nn.PReLU()]  # c_in->c_ft
        layers += [nn.Conv2d(n_feats, pred_feats, 3, stride=1, padding=1)]  # c_ft->c_red
        for _ in range(n_layers - 2):
            layers += [nn.PReLU(), nn.Conv2d(pred_feats, pred_feats, 3, stride=1, padding=1)]  # (c_red->c_red)x(N-2)
        layers += [nn.PReLU(), nn.Conv2d(pred_feats, nc_adapter, 3, stride=1, padding=1)]  # c_red->c_ada
        self.conv = nn.Sequential(*layers)

    def forward(self, x, depth=None):
        x = self.conv(x)
        if depth is not None:
            x = x * depth.view(-1, 1, 1, 1)  # (b,1)->(b,1,1,1), [0,1]*depth
        return x.clamp(0, self.upper_bound)  # clamp to (0,max)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, conv, n_feats, kernel_size, clamp=None, bias=True,
                 bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        self.clamp = clamp
        self.res_scale = res_scale
        # conv+relu+conv
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=kernel_size // 2))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x, attention=None):

        if attention is None:
            d_map = torch.ones((1, 1, *x.shape[-2:]), device=x.device)  # d_map = I
        else:
            d_map = self.clamp(attention)  # d_map

        res = self.body(x).mul(self.res_scale)
        res = res * d_map
        return res + x


if __name__ == "__main__":
    print(__file__)
