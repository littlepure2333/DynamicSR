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
        self.down = common.DownBlock(scale)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors*(scale**2), n_feats, k)]

        m_body = [common.ResBlock(conv, n_feats, k, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, k))

        m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, 3, k)]

        # m_tail = [common.CARAFEUpsampler(conv=None, n_feats=n_feats, scale_max=scale, up_kernel=5,
        #                                  compressed_channels=64),
        #           nn.Conv2d(n_feats, 3, 3, 1, 1)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, krl=None):
        x = self.sub_mean(x)
        x = self.down(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


if __name__ == '__main__':
    print(__file__)

