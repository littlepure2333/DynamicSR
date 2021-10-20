from model import common
import torch.nn as nn
import copy
import torchsnooper

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
        exit_interval = args.exit_interval
        self.exit_list = list(range(exit_interval-1, n_resblocks, exit_interval))
        self.exit_index = len(self.exit_list) - 1
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

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if args.shared_tail:
            self.tail = nn.Sequential(*m_tail)
            self.forward_tail = self.tail
        else:
            self.tail = nn.ModuleList([copy.deepcopy(nn.Sequential(*m_tail)) for _ in self.exit_list])
            self.forward_tail = self.tail[self.exit_index]

    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        forward_body = self.body[:self.exit_list[self.exit_index]+1]
        res = forward_body(x)
        res += x

        x = self.forward_tail(x)
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