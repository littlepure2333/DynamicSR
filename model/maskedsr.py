import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from model import common
from model import maskutil

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

        self.m_head = conv(args.n_colors, n_feats, k)  # 3->n_feats

        self.m_body = nn.ModuleList()
        for _ in range(n_resblocks):
            self.m_body.append(maskutil.ResBlock(conv, n_feats, k, act=act, res_scale=args.res_scale))  # maskconv x N

        self.m_body_p = conv(n_feats, n_feats, k)  # n_feats->n_feats
        self.m_tail = nn.Sequential(*[common.Upsampler(conv, scale, n_feats, act=False), conv(n_feats, 3, k)])  # ->3

    def forward(self, x, krl=None, meta=None):
        x = self.sub_mean(x)  # sub mean
        x = self.m_head(x)  # 3->n_feats
        res, meta = self.m_body[0]((x, meta))  # n_feats->n_feats
        for k in range(1, self.n_resblocks):
            res, meta = self.m_body[k]((res, meta))  # n_feats->n_feats
        res = self.m_body_p(res)  # n_feats->n_feats
        res += x
        x = self.m_tail(res)
        x = self.add_mean(x)
        return x, meta


if __name__ == '__main__':
    print(__file__)
