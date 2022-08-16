import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fanjiang.layers import trunc_normal_, c2_msra_fill, constant_fill, Conv2d
from torch.distributions import Normal
from torch.cuda.amp import autocast
from ..transformer import SwinBlock

__all__ = ["GaussianLayer", "GaussianPerturbation"]


class GaussianLayer(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            image_size,
            patch_size,
            window_size,
            depth=4,
            num_heads=8,
            mlp_ratio=4.,
            init_mean=None,
            init_var=None,
        ):

        super().__init__()
        self.depth = depth
        self.input_size = [sz // patch_size for sz in image_size]

        self.down_layer = nn.Conv2d(
            in_channels, channels, kernel_size=patch_size, stride=patch_size
        )
        c2_msra_fill(self.down_layer)

        self.linear_proj = nn.Linear(2 * channels, channels, bias=False)

        self.blocks = nn.ModuleList()
        for i in range(depth):

            blk = SwinBlock(
                dim=channels,
                input_size=self.input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            )

            self.blocks.append(blk)

        self.conv_proj = Conv2d(
            channels * depth, channels, kernel_size=3, padding=1,
            activation=F.gelu,
        )

        self.mean = nn.Conv2d(
            channels, in_channels, kernel_size=3, padding=1
        )
        if init_mean is not None:
            constant_fill(self.mean, init_mean)

        self.var = nn.Conv2d(
            channels, in_channels, kernel_size=3, padding=1
        )
        if init_var is not None:
            constant_fill(self.var, init_var)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input, state, emb=None):
        x = self.down_layer(input)

        state = torch.cat([x, state], dim=1)
        state =  self.linear_proj(rearrange(state, 'n c h w -> n (h w) c'))

        if emb is not None:
            state = state + emb.unsqueeze(0).unsqueeze(0)

        states = []
        for blk in self.blocks:
            state = blk(state)
            states.append(state)

        state = torch.cat(states, dim=-1)
        state = rearrange(state,  'n (h w) c -> n c h w', h=self.input_size[0])

        with autocast(enabled=False):
            state = self.conv_proj(state.float())
            mean = self.mean(state)
            var = self.var(state).mul(0.5).exp()

        dist = Normal(mean, scale=var)
        return dist, state



class GaussianPerturbation(nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            embed_dim,
            image_size,
            patch_size,
            window_size,
            depth=4,
            mlp_ratio=4.,
        ):
        super().__init__()

        self.prior = GaussianLayer(
            in_channels,
            channels,
            image_size,
            patch_size,
            window_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            #init_mean=0,
            #init_var=0,
        )

        self.posterior = GaussianLayer(
            in_channels,
            channels,
            image_size,
            patch_size,
            window_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )

        #self.conv = nn.Conv2d(embed_dim, channels, kernel_size=3, padding=1)

    def init_state(self, state):
        #state = self.conv(state)
        state_p = state.clone()
        state_q = state.clone()
        return state_p, state_q






