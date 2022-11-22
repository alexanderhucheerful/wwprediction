import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.ops import DeformConv2dPack as DCN

from .batch_norm import get_norm
from .weight_init import c2_msra_fill
from .wrappers import Conv2d


class LKBlock(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.norm = get_norm("GN", d_model, groups=1)
        self.pw1 = Conv2d(d_model, d_model, 1)
        self.dwconv = Conv2d(d_model, d_model, 7, padding=3, groups=d_model)
        self.pw2 = Conv2d(d_model, d_model, 1)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.norm(x)
        out = self.pw1(out)
        out = self.dwconv(out)
        out = self.act(out)
        out = self.pw2(out)
        return x + out

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm="BN"):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        for layer in [self.conv1, self.conv2]:
            c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm="BN"):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        c2_msra_fill(self.conv)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        out = F.relu_(out)
        return out


class LBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=1,
        norm=None,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels - in_channels),
        )

        self.conv2 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, out_channels),
        )

        self.conv3 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3]:
            c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        shortcut = torch.cat([x, out], dim=1)

        out = self.conv2(x)
        out = F.relu_(out)
        out = self.conv3(out)

        out += shortcut
        out = F.relu_(out)
        return out


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=1,
        norm=None,
        dilation=1,
    ):
        super().__init__()

        self.shortcut = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu_(out)
        return out


class DcnBlock(nn.Module):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_num_groups=1,
    ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = DCN(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
        )
        self.norm = get_norm(norm, bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.norm(out)

        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

