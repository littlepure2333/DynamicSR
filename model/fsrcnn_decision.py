import torch.nn as nn
import torch

from model import common

def make_model(args, parent=False):
    return FSRCNN(args)

class FSRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()

        scale = args.scale[0]

        self.test_only = args.test_only
        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold

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

        # define early-exiting decision-maker
        m_eedm = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(s,1),
            nn.Tanh()
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.eedm = nn.Sequential(*m_eedm)

    def forward(self, x):
        if not self.test_only: # training mode / eval mode
            fea = self.head(x)

            outputs = []
            decisions = []
            for i, layer in enumerate(self.body):
                fea = layer(fea)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.tail(fea)
                    decision = self.eedm(fea)
                    outputs.append(output)
                    decisions.append(decision)
            return outputs, decisions
        else: # test mode
            fea = self.head(x)

            for i, layer in enumerate(self.body):
                fea = layer(fea)
                if i % self.exit_interval == (self.exit_interval-1):
                    output = self.tail(fea)
                    decision = self.eedm(fea)
                    if decision >= self.exit_threshold:
                        return output, (i-(self.exit_interval-1))//self.exit_interval, decision
            return output, (i-(self.exit_interval-1))//self.exit_interval, decision
