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
        act = nn.ReLU(True)
        self.args = args
        self.scale = args.scale[0]  # [2,3,4,5..]
        self.scale_idx = 0  # scale idx
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
                  for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        self.upsampler = common.CARAFEUpsampler(conv=None, n_feats=n_feats, scale_max=4)
        self.last_conv = nn.Conv2d(n_feats, 3, 3, 1, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x, krl=None):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.upsampler(res, self.scale)
        x = self.last_conv(x)
        x = self.add_mean(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx  # scale idx
        self.scale = self.args.scale[scale_idx]  # scale

# class EDSRSFT(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(EDSRSFT, self).__init__()
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3
#         scale = args.scale[0]
#         act = nn.ReLU(True)
#         self.sub_mean = common.MeanShift(args.rgb_range)
#         self.add_mean = common.MeanShift(args.rgb_range, sign=1)
#         m_head = [conv(args.n_colors, n_feats, kernel_size)]  # 3->n_feats
#
#         m_body = [common.SFTResBlock(nf=n_feats,para=21) for _ in range(n_resblocks)]
#         m_body.append(common.SFTBasic(nf=n_feats,para=21))
#
#         m_body.append(nn.Conv2d(n_feats,n_feats,3,1,1))
#         m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
#                   conv(n_feats, args.n_colors, kernel_size)]
#
#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)
#
#     def forward(self, x, krl=None):
#         B, C, H, W = x.size()  # (B,C,H,W)
#         B_h, C_h = krl.size()  # (B,L)
#         krl_map = krl.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))
#         x = self.sub_mean(x)
#         fea = self.head(x)
#         res = self.body((fea,krl_map))
#         fea = fea + res
#         x = self.tail(fea)
#         x = self.add_mean(x)
#         return x
