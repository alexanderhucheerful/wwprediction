import torch 
import torch.nn as nn
import torch.nn.functional as F
from fanjiang.builder import MODELS
from fanjiang.layers import Conv2d
from fanjiang.layers.norm import get_norm
from torch.nn.utils import spectral_norm
from fanjiang.layers import HaarTransform

@MODELS.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        in_channels (int): Channel number of inputs. Default: 3.
        channels (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, in_channels, channels=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(channels * 8, channels * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(channels * 4, channels * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

@MODELS.register()
class PixelDiscriminator(nn.Module):

    def __init__(
        self, 
        in_channels, 
        channels=32, 
        num_layers=3, 
        norm="IN", 
        **kwargs,
    ):
        super().__init__()
        self.dwt = HaarTransform(in_channels)

        in_channels *= 4
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if norm == "SN":
                layer = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
                    ),
                    nn.LeakyReLU(0.01)
                )
            else:
                layer = Conv2d(
                    in_channels, 
                    channels, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False,
                    norm=get_norm(norm, channels) if i > 0 else None,
                    activation=nn.LeakyReLU(0.2),
                )
            self.layers.append(layer)

            in_channels = channels
            channels = channels * 2

        self.final_layer = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.dwt(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


@MODELS.register()
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, 
        in_channels, 
        channels=64, 
        num_layers=3, 
        norm='IN', 
    ):
        super().__init__()
    
        kw = 4
        padw = 1
        use_bias = norm == "IN"

        if norm == 'SN':
            sequence = [
                spectral_norm(
                    nn.Conv2d(in_channels,
                              channels,
                              kernel_size=kw,
                              stride=2,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
            sequence = [
                nn.Conv2d(in_channels,
                          channels,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias),
                nn.LeakyReLU(0.2)
            ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm == 'SN':
                sequence += [
                    spectral_norm(
                        nn.Conv2d(channels * nf_mult_prev,
                                  channels * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw)),
                    nn.LeakyReLU(0.01)
                ]
            else:
                sequence += [
                    nn.Conv2d(channels * nf_mult_prev,
                              channels * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias=use_bias),
                    get_norm(norm, channels * nf_mult),
                    nn.LeakyReLU(0.2)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, 8)
        if norm == 'SN':
            sequence += [
                spectral_norm(
                    nn.Conv2d(channels * nf_mult_prev,
                              channels * nf_mult,
                              kernel_size=kw,
                              stride=1,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
            sequence += [
                nn.Conv2d(channels * nf_mult_prev,
                          channels * nf_mult,
                          kernel_size=kw,
                          stride=1,
                          padding=padw,
                          bias=use_bias),
                get_norm(norm, channels * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        if norm == 'SN':
            sequence += [
                spectral_norm(
                    nn.Conv2d(channels * nf_mult,
                              1,
                              kernel_size=kw,
                              stride=1,
                              padding=padw,
                              bias=False))
            ]  # output 1 channel prediction map
        else:
            sequence += [
                nn.Conv2d(channels * nf_mult,
                          1,
                          kernel_size=kw,
                          stride=1,
                          padding=padw)
            ]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


@MODELS.register()
class MixDiscriminator(nn.Module):

    def __init__(
        self, 
        num_frames,
        num_levels,  
        channels=32, 
        num_layers=3, 
        norm="IN", 
        **kwargs,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_levels = num_levels
        self.disc_t = PixelDiscriminator(num_frames, channels, num_layers, norm)
        self.disc_c = PixelDiscriminator(num_levels, channels, num_layers, norm)
    
    def forward(self, x):
        xt, xc = x.split([self.num_frames, self.num_levels], dim=1)
        xt = self.disc_t(xt)
        xc = self.disc_c(xc)
        out = torch.cat([xt, xc], dim=1)
        return out
