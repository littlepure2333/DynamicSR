import torch.nn as nn
import torch

from model import common

def make_model(args, parent=False):
    return FSRCNN(args)

class FSRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()

        scale = args.scale[0]

        # define head module
        m_head = [
            common.BasicBlock(conv, args.n_colors, d, kernel_size=5, bn=False, act=nn.PReLU(d)),
            common.BasicBlock(conv, d, s, kernel_size=1, bn=False, act=nn.PReLU(s)),
        ]

        # define body module
        m_body = [
            common.BasicBlock(conv, s, s, kernel_size=3, bn=False, act=nn.PReLU(s)) for _ in range(m)
        ]

        # define tail module
        m_tail = [
            common.BasicBlock(conv, s, d, kernel_size=1, bn=False, act=nn.PReLU(d)),
            nn.ConvTranspose2d(in_channels=d, out_channels=args.n_colors, kernel_size=9,
                                stride=scale, padding=9//2, output_padding=scale-1)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        fea = self.head(x)
        fea = self.body(fea)
        out = self.tail(fea)
        return out