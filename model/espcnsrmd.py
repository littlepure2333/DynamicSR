import torch
import torch.nn as nn
import torch.nn.init as init
import cv2
import numpy as np
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


class ESPCN(nn.Module):
    """ESPCN with SRMD"""
    
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        self.scale = args.scale[0]
        upscale_factor = args.scale[0]
        n_colors = args.n_colors

        srmd_dim = 21
        self.conv1 = nn.Conv2d(n_colors+srmd_dim, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x, krl=None):
        B, C, H, W = x.size()  # (B,C,H,W)
        B_h,C_h = krl.size()
        krl_map = krl.view((B_h,C_h,1,1)).expand((B_h,C_h,H,W))  # kernel map
        x_srmd = torch.cat((x,krl_map),dim=1)

        out = self.act_func(self.conv1(x_srmd))
        out = self.act_func(self.conv2(out))
        out = self.act_func(self.conv3(out))
        out = self.pixel_shuffle(self.conv4(out))
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


