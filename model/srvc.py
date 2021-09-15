import math
from model import common
import torchsnooper

import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return SRVC(args)

class SRVC(nn.Module):
    def __init__(self, args):
        super(SRVC, self).__init__()
        self.f = args.f
        self.F = args.F
        self.n_feats = args.n_feats # original 128
        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        scale = args.scale[0]
        self.scale = scale

        self.s2b = common.Space2Batch(self.patch_h)

        # define head
        # self.head = common.BasicBlock(common.valid_conv, 3, self.f, 3, bias=True, bn=False)
        m_head = [common.BasicBlock(common.valid_conv, 3, self.f, 3, bias=True, bn=False),
                  common.BasicBlock(common.valid_conv, self.f, self.f, 3, bias=True, bn=False)]
        self.head = nn.Sequential(*m_head)
        
        # define adaConv
        self.kernel = common.BasicBlock(common.valid_conv, self.f, 27*self.F, 3, bias=True, bn=False, act=None)
        self.bias = common.BasicBlock(common.valid_conv, self.f, self.F, 3, bias=True, bn=False, act=None)
        # self.adaConv = common.AdaConv(3)

        self.b2s = common.Batch2Space(self.patch_h)
        
        # define body
        # m_body = [common.MyResBlock(common.default_conv, self.F, self.n_feats, 5, bias=True, bn=False),
        #           common.MyResBlock(common.default_conv, self.n_feats, 32, 3, bias=True, bn=False),
        #           common.MyResBlock(common.default_conv, 32, 3*scale*scale, 3, bias=True, bn=False, act=None)]

        m_body = [common.MyResBlock(common.default_conv, self.F, self.n_feats, 5, bias=True, bn=False),
                  common.MyResBlock(common.default_conv, self.n_feats, 3*scale*scale, 3, bias=True, bn=False, act=None)]
        self.body = nn.Sequential(*m_body)

        # define tail
        self.tail = nn.PixelShuffle(scale)
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    # @torchsnooper.snoop()
    def forward(self, x):
        # RGB mean for DIV2K
        x = self.sub_mean(x)

        # full -> 5*5 (including padding)
        x, B,C,H,W,pad_h,pad_w = self.s2b(x)

        # 5*5 -> 3*3
        feature = self.head(x)

        # 3*3 -> 1*1
        kernel = self.kernel(feature).view(-1, self.F, 3, 3, 3) # (B, out_C, in_C, iH, iW)
        bias = self.bias(feature).view(-1, self.F) # (B, out_C)

        # 5*5 -> 5*5
        x = common.adaConv(x, kernel, bias)

        # 5*5 -> full
        x = self.b2s(x,B,C,H+pad_h,W+pad_w)

        x = self.body(x)
        x = self.tail(x)

        # unpadding
        x = x[...,:H*self.scale,:W*self.scale]

        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))