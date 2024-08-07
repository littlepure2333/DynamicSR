from model import common

import torch
import torch.nn as nn
import utility

def make_model(args, parent=False):
    return VDSR(args)

class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.scale = args.scale[0]
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)

        self.args = args
        self.test_only = args.test_only
        self.exit_interval = args.exit_interval
        self.exit_threshold = args.exit_threshold

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, args.n_colors, None))

        # define early-exiting decision-maker
        m_eedm = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feats,1),
            nn.Tanh()
        ]

        self.body = nn.Sequential(*m_body)
        self.eedm = nn.Sequential(*m_eedm)

    def forward(self, input):
        x, hr = input
        x = self.upsample(x)
        x = self.sub_mean(x)
        res = self.body[0](x)

        exit_index = torch.ones(x.shape[0],device=x.device) * (-1.)
        pass_index = torch.arange(0,x.shape[0],device=x.device)
        pre_psnr = 0
        for i, layer in enumerate(self.body[1:-1]):
            res[pass_index,...] = layer(res[pass_index,...])
            if i % self.exit_interval == (self.exit_interval-1):
                sr_i = self.add_mean(x + self.body[-1](res))
                now_psnr = utility.calc_psnr(sr_i, hr, self.scale, self.args.rgb_range)
                decision = 1 - torch.tanh(torch.tensor(now_psnr - pre_psnr))
                pre_psnr = now_psnr
                # decision = self.eedm(res)
                pass_index = torch.where(decision<self.exit_threshold)[0]

                remain_id = torch.where(exit_index < 0.0)[0]
                exit_id = torch.where(decision>=self.exit_threshold)[0]
                intersection_id = []
                for id in exit_id:
                    if id in remain_id: intersection_id.append(int(id))
                exit_index[intersection_id] = (i-(self.exit_interval-1))//self.exit_interval

        output = self.add_mean(x + self.body[-1](res))
        return output, exit_index, decision


