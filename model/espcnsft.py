import torch
import torch.nn as nn
import torch.nn.init as init
import cv2
import numpy as np
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


class ESPCN(nn.Module):
    """ESPCN Spatial Feature Transform"""
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        upscale_factor = args.scale[0]  # upscale factor
        n_colors = args.n_colors

        self.conv1 = nn.Conv2d(n_colors, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))

        self.sft1 = common.SFTBasic(nf=64, para=21, idx=1)
        self.sft2 = common.SFTBasic(nf=64, para=21, idx=2)
        self.sft3 = common.SFTBasic(nf=32, para=21, idx=3)

        # self.sft1 = common.SFTGhost(nf=64, para=21, idx=1)
        # self.sft2 = common.SFTGhost(nf=64, para=21, idx=2)
        # self.sft3 = common.SFTGhost(nf=32, para=21, idx=3)
        #
        # self.sft1 = common.SFTDP(nf=64, para=21, idx=1)
        # self.sft2 = common.SFTDP(nf=64, para=21, idx=2)
        # self.sft3 = common.SFTDP(nf=32, para=21, idx=3)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x, krl=None):
        B, C, H, W = x.size()  # (B,C,H,W)
        B_h, C_h = krl.size()  # (B,L)
        krl_map = krl.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel map

        x = self.act_func(self.sft1((self.conv1(x), krl_map)))
        x = self.act_func(self.sft2((self.conv2(x), krl_map)))
        x = self.act_func(self.sft3((self.conv3(x), krl_map)))
        x = self.pixel_shuffle(self.conv4(x))
        return x


##########################
# class ESPCNSFT(nn.Module):
#     def __init__(self, args):
#         super(ESPCNSFT, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         self.conv1 = nn.Conv2d(n_colors, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(64, n_colors * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
#
#         self.sft_base = common.SFTBasic(nf=64, para=21, idx=0)
#         self.sft1 = nn.Conv2d(64,64,kernel_size=3,stride=1,groups=64)
#         self.sft2 = nn.Conv2d(64,64,kernel_size=3,stride=1,groups=64)
#         self.sft3 = nn.Conv2d(64,64,kernel_size=3,stride=1,groups=64)
#
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#
#     def forward(self, x, krl=None):
#         B, C, H, W = x.size()  # (B,C,H,W)
#         B_h, C_h = krl.size()  # (B,L)
#         krl_map = krl.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel map
#
#         x = self.act_func(self.sft1(self.sft_base((self.conv1(x), krl_map))))
#         x = self.act_func(self.sft2(self.sft_base((self.conv2(x), krl_map))))
#         x = self.act_func(self.sft3(self.sft_base((self.conv3(x), krl_map))))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x

