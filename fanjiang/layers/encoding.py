import math
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["TimeEncoding", "ConvPosEncoding"]

class ConvPosEncoding(nn.Module):
    def __init__(self, dim, k=3, act=False):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x


class TimeEncoding(nn.Module):
    def __init__(self, dim, max_period=1e4, drop=0):
        super().__init__()
        self.num_freq = dim // 2
        self.max_period = max_period
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        emb = math.log(self.max_period) / (self.num_freq - 1)
        emb = torch.exp(torch.arange(self.num_freq, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = self.act(self.fc1(emb))
        emb = self.drop(emb)
        emb = self.fc2(emb)
        emb = self.drop(emb)
        return emb


