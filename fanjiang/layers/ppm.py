import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import get_norm
from .wrappers import Conv2d


class PPM(nn.Module):
    def __init__(
            self, 
            in_channels, 
            channels, 
            pool_scales=(1, 2, 3, 6), 
            norm=None,
            act=None,
        ):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels

        self.layers = nn.ModuleList()
        for pool_scale in pool_scales:
            self.layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    Conv2d(
                        self.in_channels,
                        self.channels,
                        1,
                        bias=False,
                        norm=get_norm(norm, channels),
                        activation=act,
                    )))
        
        self.bottleneck = Conv2d(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            norm=get_norm(norm, channels), 
            activation=act,
        )

    
    def forward(self, x):
        """Forward function."""
        ppm_outs = [x]
        for ppm in self.layers:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)
            ppm_outs.append(upsampled_ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        out = self.bottleneck(ppm_outs)
        return out


