import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ecb import ECB
# from ecb import ECB

class ECBSR(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(ECBSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        backbone += [ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        # y = self.backbone(x) + x
        y = self.backbone(x)
        y = y + x
        y = self.upsampler(y)
        return y

if __name__ == "__main__":
    # RGB
    # input = torch.ones((1,3,20,20))
    # model = ECBSR(4,32,4,'prelu',2,3)
    # Y
    input = torch.ones((1,1,20,20))
    model = ECBSR(4,32,4,'prelu',2,1)
    output = model(input)