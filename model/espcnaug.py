import torch
import torch.nn as nn
import torch.nn.init as init
import cv2
import numpy as np
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


class ESPCN(nn.Module):
    """ESPCN baseline"""
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])  # use scale[0]
        self.n_colors = args.n_colors
        self.down = common.DownBlock(self.scale)
        self.conv1 = nn.Conv2d(self.n_colors*(self.scale**2), 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        self._initialize_weights()

    def forward(self, x, krl=None):
        # (x, krl)
        out = self.act_func(self.conv1(self.down(x)))
        out = self.act_func(self.conv2(out))
        out = self.act_func(self.conv3(out))
        out = self.pixel_shuffle(self.conv4(out))
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


if __name__ == '__main__':
    pass



