import torch.nn as nn
import torch
from torch.nn import Upsample
from .ops_dcnv3.modules import DCNv3

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class BottleneckCSP(nn.Module):
# CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        
    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


def forward(self, x):
    y1 = self.cv3(self.m(self.cv1(x)))
    y2 = self.cv2(x)
    return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Segment(nn.Module):
    def __init__(self, in_channel=256, classes=1):
        super(Segment, self).__init__()
        self.layers = nn.Sequential(
            Conv(in_channel, 128, 3, 1),
            Upsample(scale_factor=2, mode='nearest'),
            BottleneckCSP(128, 64, 1, False),
            Conv(64, 32, 3, 1),
            Upsample(scale_factor=2, mode='nearest'),
            Conv(32, 16, 3, 1),
            BottleneckCSP(16, 8, 1, False),
            Upsample(scale_factor=2, mode='nearest'),
            Conv(8, classes, 3, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        # print(out)
        return out
    
# class Segment_DCN(nn.Module):
#     def __init__(self, in_channel=256, classes=1):
#         super(Segment_DCN, self).__init__()

#         self.layers = nn.Sequential(
#             DCNv3(128, 3, 1),
#             Upsample(scale_factor=2, mode='nearest'),
#             BottleneckCSP(128, 64, 1, False),
#             DCNv3(32, 3, 1),
#             Upsample(scale_factor=2, mode='nearest'),
#             DCNv3(16, 3, 1),
#             BottleneckCSP(16, 8, 1, False),
#             Upsample(scale_factor=2, mode='nearest'),
#             Conv(8, classes, 3, 1),
#         )

#     def forward(self, x):
#         return self.layers(x)