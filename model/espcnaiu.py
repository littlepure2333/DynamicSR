import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


class ESPCN(nn.Module):
    """ESPCN Arbitrary Integral Upsampler (AIU)"""
    
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.args = args
        self.scale = args.scale[0]  # [2,3,4,5..]
        self.scale_idx = 0  # scale idx
        self.act_func = nn.ReLU(inplace=True)
        n_colors = args.n_colors

        self.conv1 = nn.Conv2d(n_colors, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))

        self.pixel_shuffle = common.CARAFEUpsampler(conv=None, n_feats=32, scale_max=4, compressed_channels=8)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1)
        self._initialize_weights()

    def forward(self, x, krl=None):
        out = self.act_func(self.conv1(x))
        out = self.act_func(self.conv2(out))
        out = self.act_func(self.conv3(out))

        out = self.pixel_shuffle(out, self.scale)  # (out, scale)
        # out = self.act_func(self.conv4(out))  # act func <-
        out = self.conv4(out)

        # out = self.pixel_shuffle(self.conv4(out))

        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx  # scale idx
        self.scale = self.args.scale[scale_idx]  # scale


if __name__ == '__main__':
    pass

