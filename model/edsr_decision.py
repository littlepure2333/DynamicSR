import math
from model import common

import torch.nn as nn

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

        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # define early-exiting decision-maker
        m_eedm = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feats,1),
            nn.Sigmoid()
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.eedm = nn.Sequential(*m_eedm)

    def forward(self, x):
        if self.training: # training mode
            x = self.sub_mean(x)
            x = self.head(x)
            res = x

            outputs = []
            decisions = []
            for i, layer in enumerate(self.body):
                res = layer(res)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.add_mean(self.tail(x + res))
                    decision = self.eedm(res)
                    outputs.append(output)
                    decisions.append(decision)
                    # output.append(self.add_mean(self.tail(x + res)))

            # x = self.tail(res)
            # x = self.add_mean(x)

            return outputs, decisions
        else: # evaluate mode
            x = self.sub_mean(x)
            x = self.head(x)
            res = x

            for i, layer in enumerate(self.body):
                res = layer(res)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.add_mean(self.tail(x + res))
                    decision = self.eedm(res)
                    if decision >= self.exit_threshold:
                        return output, i, decision
            return output, i, decision


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