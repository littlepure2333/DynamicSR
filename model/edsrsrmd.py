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
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        srmd_dim=21
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors+srmd_dim, n_feats, kernel_size)]
        # define body module
        m_body = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
                  for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, krl=None):
        B, C, H, W = x.size()  # (B,C,H,W)
        B_h, C_h = krl.size()  # (B,L)
        krl_map = krl.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))
        x = self.sub_mean(x)
        x_srmd = torch.cat((x,krl_map),dim=1)
        x = self.head(x_srmd)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


if __name__ == '__main__':
    print(__file__)