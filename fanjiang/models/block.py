import torch
import torch.nn as nn
from einops import rearrange
from fanjiang.layers import (Attention, ChannelAttention, GlobalFilter, ConvPosEncoding,
                             DropPath, c2_msra_fill, get_norm, to_2tuple)



def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            c2_msra_fill(layer)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()

        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()


    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        out = x + out
        return out


class GFBlock(nn.Module):
    def __init__(self, dim, input_size, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_value=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, input_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) if init_value > 0 else None


    def forward(self, x):
        out = x
        x = self.mlp(self.norm2(self.filter(self.norm1(x))))

        if self.gamma is not None:
            x = self.gamma * x

        out = self.drop_path(x) + out
        return out


class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, input_size, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        self.input_size = input_size

        self.cpe = nn.ModuleList([ConvPosEncoding(dim=dim, k=3, act=cpe_act),
                                  ConvPosEncoding(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)


    def forward(self, x):
        x = self.cpe[0](x, self.input_size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, self.input_size)

        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_size,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_size = input_size
        self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.cpe = nn.ModuleList([ConvPosEncoding(dim=dim, k=3),
                                  ConvPosEncoding(dim=dim, k=3)])

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.proj = nn.Linear(in_channels + channels, channels, bias=False)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, state=None):
        if self.cond_proj:
            x = torch.cat([x, state], dim=1)

        B, L, C = x.shape
        shortcut = self.cpe[0](x, self.input_size)

        x = self.norm1(shortcut)
        x = x.view(B, *self.input_size, C)

        if max(self.window_size) > 0:
            x = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size[0] * self.window_size[1], C)

        x = self.attn(x)
        x = x.view(-1, self.window_size[0], self.window_size[1], C)

        if max(self.window_size) > 0:
            x = window_reverse(x, self.window_size, *self.input_size)

        x = x.view(B, L, C)
        x = shortcut + self.drop_path(x)
        x = self.cpe[1](x, self.input_size)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



