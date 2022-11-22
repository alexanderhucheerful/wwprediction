# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, **kwargs)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        mode = kwargs.pop("mode", None)
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"


        if self.mode is not None:
            x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode=self.mode[1])
            x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode=self.mode[0])
            x = F.conv2d(
                x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
            )
        else:
            x = F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        # x = F.conv2d(
        #     x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        # )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# class AdaIN(nn.Module):
#     def __init__(self, style_channels, channels):
#         super().__init__()
#         self.norm = nn.InstanceNorm1d(channels)
#         self.affine = EqualLinear(style_channels, channels * 2)
#         self.affine.bias.data[:channels] = 1
#         self.affine.bias.data[channels:] = 0

#     # def forward(self, x, style):
#     #     style = self.affine(style).unsqueeze(2).unsqueeze(3)
#     #     gamma, beta = style.chunk(2, 1)
#     #     out = self.norm(x)
#     #     out = gamma * out + beta
#     #     return rearrange(out, 'n c h w -> n (h w) c')

#     def forward(self, x, ws):
#         ws = self.affine(ws).unsqueeze(1)
#         gamma, beta = ws.chunk(2, 2)
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         x = x.transpose(1, 2)
#         out = gamma * x + beta
#         return out


class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_channels,
        kernel_size=3,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        fan_in = in_channels * (kernel_size * kernel_size)
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.affine = EqualLinear(style_channels, in_channels, bias_init=1)


    def forward(self, x, style):
        bs = len(style)
        style = self.affine(style).reshape((bs, 1, -1, 1, 1))
        weight = self.scale * self.weight * style
        demod = torch.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.reshape((bs, -1, 1, 1, 1))
        weight = weight.flatten(0, 1)
        out = F.conv2d(x, weight, padding=self.padding, groups=bs)
        return rearrange(out, '1 (n c) h w -> n (h w) c', n=bs)



ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate
Linear = torch.nn.Linear


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]

