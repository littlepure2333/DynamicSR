from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return PatchNet(args)

class PatchNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(PatchNet, self).__init__()

        act = nn.LeakyReLU(0.2, True)
        in_channels = args.n_colors
        out_channels= None

        # define body module
        m_body = []
        layers = ['64*1','64*1','128*1','128*1','256*1','256*1']

        for l in layers:
            out_channels, number = [int(i) for i in l.split('*')]
            for i in range(number):
                m_body.append(common.Bottleneck(in_channels, out_channels, act=act))
            in_channels = out_channels
            m_body.append(nn.AvgPool2d(2, stride=2))

        # define tail module
        m_tail = [
            nn.Conv2d(in_channels, 1, 1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)

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