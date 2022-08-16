import torch.nn.functional as F
from einops.einops import rearrange
from torch import nn as nn

from .helpers import to_2tuple, to_3tuple
from .norm import get_norm


class OverlapedPatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patch = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



class ResEmbed(nn.Module):
    def __init__(self,
            input_frames=1,
            image_size=224,
            patch_size=4,
            in_channels=3,
            embed_dim=96,
            norm_layer=None,
        ):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        input_size = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]

        self.image_size = image_size
        self.patch_size = patch_size
        self.input_size = input_size

        self.num_patch = input_size[0] * input_size[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.input_frames = input_frames

        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=(patch_size[0], patch_size[1]),
                stride=(patch_size[0], patch_size[1]),
                padding=(0, 0),
            ),
            # get_norm("LN", embed_dim),
        )

        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.proj = nn.Conv3d(
            embed_dim, embed_dim,
            kernel_size=(input_frames, 1, 1),
            stride=(input_frames, 1, 1),
            padding=0
        )


    def forward(self, x):
        x = self.down(x.flatten(0, 1))

        residual = x
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x += residual
        x = F.gelu(x)

        x = rearrange(x, '(n t) c h w -> n c t h w', t=self.input_frames)
        x = self.proj(x)
        x = rearrange(x, 'n c t h w -> n (t h w) c')
        return x

    def flops(self):
        flops = 0
        #self.down
        flops += self.num_patch * self.patch_size[0] * self.patch_size[1] * self.in_channels * self.embed_dim
        # res conv
        flops += 2 * self.num_patch * 3 * 3 * self.embed_dim * self.embed_dim
        #self.proj
        flops += self.num_patch * self.input_frames * self.embed_dim * self.embed_dim
        return flops
