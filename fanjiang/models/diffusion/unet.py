import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fanjiang.layers import (
    ConvResample,
    Attention,
    TimeEncoding,
    xavier_fill,
    ddpm_fill,
    Dropout,
    PixelNorm,
    AdaptiveGroupNorm
)

from fairscale.nn.checkpoint import checkpoint_wrapper

__all__ = ["Unet", "UnetSR", "UnetBaseline"]


def plot_dropout(imgs, save_dir="log_images"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    eval_names = ["z500", "t850", "t2m", "u10", "v10", "tp"]

    if torch.is_tensor(imgs):
    	imgs = imgs.cpu().numpy()

    for i, name in enumerate(eval_names):
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(imgs[i])
        save_f = os.path.join(save_dir, f"{name}.png")
        plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, transparent='true', dpi=600)
        plt.close()


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_fill(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x



class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, **kwargs):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.attn = Attention(dim, num_heads)

        self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, dim, bias=False)
            )


    def forward(self, x):
        w = x.shape[-1]

        if x.ndim == 4:
            x = rearrange(x, 'n c h w -> n (h w) c')

        x = x + self.attn(x)
        x = x + self.ffn(x)

        if x.ndim == 3:
            x = rearrange(x, 'n (h w) c -> n c h w', w=w)

        return x

# decoder
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=32,
        norm=True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        dim_emb=0,
        skip_scale=np.sqrt(2),
    ):
        super().__init__()

        self.skip_scale = skip_scale

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_emb, dim_out * 2)
        ) if dim_emb > 0 else nn.Identity()

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, temb=None, zemb=None):
        scale_shift = None

        if temb is not None:
            temb = self.time_mlp(temb)
            temb = rearrange(temb, 'b c -> b c 1 1')
            scale_shift = temb.chunk(2, dim = 1)

        h = self.block1(x)
        h = self.block2(h, scale_shift = scale_shift)

        return (h + self.res_conv(x)) / self.skip_scale


class ResnetBlockBigGAN(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        dim_emb=0,
        dropout=0.1,
        skip_scale=np.sqrt(2),
    ):
        super().__init__()

        self.skip_scale = skip_scale

        if dim_emb > 0:
            self.time_mlp = nn.Linear(dim_emb, dim_out)
            ddpm_fill(self.time_mlp)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding = 1)
        ddpm_fill(self.conv1)

        self.gn1 = AdaptiveGroupNorm(dim, dim_emb)

        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding = 1)
        ddpm_fill(self.conv1, scale=0)

        self.gn2 = AdaptiveGroupNorm(dim_out, dim_emb)

        self.shutcut = nn.Identity()

        if dim != dim_out:
            self.shutcut = nn.Conv2d(dim, dim_out, 1)
            ddpm_fill(self.shutcut)


    def forward(self, x, temb = None, zemb=None):
        shutcut = self.shutcut(x)

        h = self.act(self.gn1(x, zemb))
        h = self.conv1(h)

        if temb is not None:
            temb = self.time_mlp(self.act(temb))
            temb = rearrange(temb, 'b c -> b c 1 1')
            h += temb

        h = self.act(self.gn2(h, zemb))
        h = self.dropout(h)
        h = self.conv2(h)

        return (shutcut + h) / self.skip_scale


class Unet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim=128,
        dim_emb=128,
        dim_mults=(1, 2, 4, 8),
        num_blocks=(2, 4, 8, 8),
        use_checkpoint=False,
        layer_attn=(False, False, False, True),
    ):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        res_class = ResnetBlock
        up_sample = lambda dim: PixelShuffleUpsample(dim, dim, 2)

        if use_checkpoint:
            attn_block = lambda dim: checkpoint_wrapper(AttentionBlock(dim))
            res_block = lambda dim, dim_out, dim_emb: checkpoint_wrapper(res_class(dim, dim_out, dim_emb))
        else:
            attn_block = lambda dim: AttentionBlock(dim)
            res_block = lambda dim, dim_out, dim_emb: res_class(dim, dim_out, dim_emb)

        self.depth = len(in_out)

        self.init_conv = nn.Conv2d(dim_in, dim, 1)

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)

            enc_proj = res_block(dim_in, dim_out, dim_emb)
            enc_cond = nn.ModuleList([
                res_block(dim_out, dim_out, dim_emb) for _ in range(num_blocks[i])
            ])
            enc_attn = attn_block(dim_out) if layer_attn[i] else nn.Identity()
            enc_down = Downsample(dim_out) if not is_last else nn.Identity()

            self.add_module(f"enc_proj{i}", enc_proj)
            self.add_module(f"enc_cond{i}", enc_cond)
            self.add_module(f"enc_attn{i}", enc_attn)
            self.add_module(f"enc_down{i}", enc_down)


        mid_dim = dims[-1]
        self.mid_block1 = res_block(mid_dim, mid_dim, dim_emb)
        self.mid_block2 = attn_block(mid_dim)
        self.mid_block3 = res_block(mid_dim, mid_dim, dim_emb)

        num_blocks = np.flip(num_blocks)
        layer_attn = np.flip(layer_attn)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):

            dec_proj = res_block(dim_out * 2, dim_in, dim_emb)
            dec_cond = nn.ModuleList([
                res_block(dim_in, dim_in, dim_emb) for _ in range(num_blocks[i])
            ])
            dec_attn = attn_block(dim_in) if layer_attn[i] else nn.Identity()
            dec_up = up_sample(dim_in)

            self.add_module(f"dec_proj{i}", dec_proj)
            self.add_module(f"dec_cond{i}", dec_cond)
            self.add_module(f"dec_attn{i}", dec_attn)
            self.add_module(f"dec_up{i}", dec_up)

        self.final_block = res_block(dim * 2, dim, dim_emb)


    def forward(self, x, t, z=None):
        x = self.init_conv(x)
        shortcut = x.clone()

        hiddens = []
        for i in range(self.depth):
            enc_proj = getattr(self, f"enc_proj{i}")
            enc_cond = getattr(self, f"enc_cond{i}")
            enc_attn = getattr(self, f"enc_attn{i}")
            enc_down = getattr(self, f"enc_down{i}")

            x = enc_proj(x, t, z)
            for layer in enc_cond:
                x = layer(x, t, z)

            x = enc_attn(x)
            hiddens.append(x)
            x = enc_down(x)

        x = self.mid_block1(x, t, z)
        x = self.mid_block2(x)
        x = self.mid_block3(x, t, z)

        for i in range(self.depth-1):
            dec_proj = getattr(self, f"dec_proj{i}")
            dec_cond = getattr(self, f"dec_cond{i}")
            dec_attn = getattr(self, f"dec_attn{i}")
            dec_up = getattr(self, f"dec_up{i}")

            x = torch.cat((x, hiddens.pop()), dim=1)

            x = dec_proj(x, t, z)
            for layer in dec_cond:
                x = layer(x, t, z)

            x = dec_attn(x)
            x = dec_up(x)

        x = torch.cat([x, shortcut], dim=1)
        return self.final_block(x, t, z)


class UnetBaseline(Unet):
    def __init__(
        self,
        dim_in,
        dim,
        patch_size,
        dim_emb=128,
        *args,
        **kwargs
    ):
        kwargs.update(dict(
            dim_in=dim,
            dim=dim,
            dim_emb=dim_emb,
            dim_mults=(1, 2, 4, 8),
            num_blocks=(2, 4, 8, 8),
            layer_attn=(False, False, False, True),
        ))
        super().__init__(*args, **kwargs)

        self.patch_embed = nn.Conv2d(dim_in, dim, kernel_size=patch_size, stride=patch_size)

        self.head = nn.Sequential(
            nn.Conv2d(dim, (patch_size ** 2) * dim, 3, padding=1),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(dim, dim_in, 3, padding=1),
        )

        self.padder = lambda x: int(np.ceil(x / 128) * 128 - x)

    def forward(self, x, lowres_x=None, lowres_t=None, hist_x=None):
        bs, _, img_h, img_w = x.shape

        lowres_x = F.interpolate(
            lowres_x,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )
        pad_h = self.padder(img_h)
        pad_w = self.padder(img_w)
        lowres_x = F.pad(lowres_x, (0, pad_w, 0, pad_h))
        lowres_x = self.patch_embed(lowres_x)

        out = super().forward(lowres_x, lowres_t)
        out = self.head(out)
        out = out[:, :, :-pad_h, :-pad_w]
        return out


class UnetSR(Unet):
    def __init__(
        self,
        dim_in,
        dim,
        patch_size,
        dim_emb=128,
        num_mlp=4,
        cond_scale=1,
        cond_prob=0,
        embed_drop=0,
        use_hist=False,
        use_checkpoint=False,
        *args,
        **kwargs
    ):
        kwargs.update(dict(
            dim_in=dim,
            dim=dim,
            dim_emb=dim_emb,
            dim_mults=(1, 2, 4, 8),
            num_blocks=(2, 4, 8, 8),
            use_checkpoint=use_checkpoint,
            layer_attn=(False, False, False, True),
        ))
        super().__init__(*args, **kwargs)

        self.cond_scale = cond_scale
        self.use_hist = use_hist

        in_chans = 2 * dim_in + use_hist * dim_in

        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

        self.head = nn.Sequential(
            nn.Conv2d(dim, (patch_size ** 2) * dim, 3, padding=1),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(dim, dim_in, 3, padding=1),
        )

        self.time_embed = TimeEncoding(dim_emb, drop=embed_drop)

        self.padder = lambda x: int(np.ceil(x / 128) * 128 - x)


    def forward(self, x, t, lowres_x=None, lowres_t=None, hist_x=None):
        bs, _, img_h, img_w = x.shape

        lowres_x = F.interpolate(
            lowres_x,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )

        if self.use_hist:
            cond_x = torch.cat([lowres_x, hist_x], dim=1)
        else:
            cond_x = lowres_x

        pad_h = self.padder(img_h)
        pad_w = self.padder(img_w)
        cond_x = F.pad(cond_x, (0, pad_w, 0, pad_h))

        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.patch_embed(torch.cat([x, cond_x], dim=1))
        t = self.time_embed(t)

        if lowres_t is not None:
            t = t + lowres_t

        out = super().forward(x, t)

        out = self.head(out)
        out = out[:, :, :-pad_h, :-pad_w]
        return out

