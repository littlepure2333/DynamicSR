import os

from data import common
from data import srdata
from data import div2k_psnr

import numpy as np

import torch
import torch.utils.data as data

class Benchmark_PSNR(div2k_psnr.DIV2K_PSNR):
    def __init__(self, args, name='', train=True, benchmark=False):
        name = name.split("_PSNR")[0]
        super(Benchmark_PSNR, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

