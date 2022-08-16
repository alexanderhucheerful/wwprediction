import torch.nn as nn

from einops import rearrange

from fanjiang.layers import PixelNorm, AdaptiveInstanceNorm, AdaptiveGroupNorm, AdaptiveGroupNorm2D, ddpm_fill

from ..block import ResBlock, GFBlock

__all__ = ["Mapping", "Mapping2D"]


class Mapping(nn.Module):
    def __init__(self,
        dim_latent,
        dim_style,
        dim_feat,
        num_layer=7,
    ):
        super().__init__()
        layers = [PixelNorm(dim=2)]
        layer = nn.Linear(dim_latent, dim_style)
        ddpm_fill(layer)
        layers.append(layer)

        for _ in range(num_layer):
            layer = nn.Linear(dim_style, dim_style)
            ddpm_fill(layer)
            layers.append(layer)
            layers.append(nn.SiLU())

        self.mapping = nn.Sequential(*layers)
        # self.adagn = AdaptiveGroupNorm(dim_feat, dim_style)

    def forward(self, latent):
        style = self.mapping(latent)
        # feat = self.adagn(feat, style)
        return style


class Mapping2D(nn.Module):
    def __init__(self,
        dim_latent,
        dim_style,
        dim_feat,
        input_size,
        num_layer=4,
    ):
        super().__init__()
        self.pn = PixelNorm(dim=1)

        self.stem = nn.Sequential(
            PixelNorm(dim=1),
            nn.Conv2d(dim_latent, dim_style, 3, padding=1)
        )

        layers = []
        for _ in range(num_layer):
            layers.append(
                # ResBlock(dim_style, dim_style, dim_style * 2),
                GFBlock(dim_style, tuple(input_size)),
            )
        self.mapping = nn.Sequential(*layers)

        self.adagn = AdaptiveGroupNorm2D(dim_feat, dim_style)

    def forward(self, latent, feat):
        x = self.stem(latent)
        x = rearrange(x, 'n c h w -> n (h w) c')
        z = self.mapping(x)
        z = rearrange(z, 'n (h w) c -> n c h w', h=latent.shape[2])
        feat = self.adagn(feat, z)
        return feat



