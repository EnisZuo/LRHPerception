import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import VarLoss, SILogLoss
########################################################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=4),
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            #ModulatedDeformConvPack(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
        )

        self.bt = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        skip = self.bt(x)
        # x = self.channel_shuffle(x, 4)
        x = self.conv1(x)

        x = self.conv2(x)
        # print('after conv2',x.shape)
        return x + skip

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape

        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = C2f(
            in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean = 1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)

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

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(C2f, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        s = c1 + c2
        self.fw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c1, kernel_size=3, padding=1))

        self.dw = nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(s, c2, kernel_size=3, padding=1))

    def forward(self, feat, depth):
        cc = torch.cat([feat, depth], 1)
        feat_new = self.fw(cc)
        depth_new = self.dw(cc)
        return feat_new, depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
                nn.Linear(c, c//4),
                nn.LeakyReLU(),
                nn.Linear(c//4, 2))

    def forward(self, x):

        x = x.squeeze(-1).squeeze(-1)
        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x


class VADepthNet(nn.Module):
    def __init__(self, max_depth=80, prior_mean=2.54, si_lambda=0.85, img_size=(480, 640), num_duo_conv=1):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        self.up_4 = Up(768 + 384, 512)
        self.up_3 = Up(512 + 192, 256)
        self.up_2 = Up(256 + 96, 64)

        self.outc = OutConv(128, 1, self.prior_mean)

        self.C2f = C2f(512, 128, n=num_duo_conv)

        self.ref_4 = Refine(512, 128)
        self.ref_3 = Refine(256, 128)
        self.ref_2 = Refine(64, 128)

        self.var_loss = VarLoss(128, 512)
        self.si_loss = SILogLoss(self.SI_loss_lambda, self.max_depth)

        self.mlayer = nn.Sequential(
                nn.AdaptiveMaxPool2d((1,1)),
                MetricLayer(768))

    def forward(self, features, gts=None):
        
        x2, x3, x4, x5 = features
        # print("x2.shape", x2.shape, "x3.shape", x3.shape, "x4.shape", x4.shape, "x5.shape", x5.shape)
        outs = {}

        metric = self.mlayer(x5)

        x = self.up_4(x5, x4)
        d = self.C2f(x)
        if gts is not None:
            var_loss = self.var_loss(x, d, gts)

        x, d  = self.ref_4(x, d)

        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up_3(x, x3)

        x, d = self.ref_3(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u3 = F.interpolate(d, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up_2(x, x2)

        x, d = self.ref_2(x, F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))

        d_u2 = F.interpolate(d, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4

        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))
        
        outs['scale_1'] = d

        if gts is not None:
            si_loss = self.si_loss(outs, gts)
            return var_loss + si_loss
        else:
            return outs['scale_1']



