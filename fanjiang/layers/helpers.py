import collections.abc
from itertools import repeat

import numpy as np
import torch


def random_crop(inputs, border=0, size=256):
    H, W = inputs.shape[-2:]
    x1 = y1 = border
    x2 = W - border
    y2 = H - border
    if y2 - y1 > size:
        x1 = np.random.randint(x1, x2 - size + 1)
        y1 = np.random.randint(y1, y2 - size + 1)
        x2 = x1 + size
        y2 = y1 + size 
    return x1, y1, x2, y2


def make_grid(shape, device, ndim=2):
    yy = torch.linspace(0, 1, steps=shape[-2], device=device)
    xx = torch.linspace(0, 1, steps=shape[-1], device=device)
 
    if ndim == 2:
        grids = torch.stack(torch.meshgrid([yy, xx]), dim=0)
        grids = grids.unsqueeze(0).repeat(shape[0], 1, 1, 1)

    elif ndim == 3:
        zz = torch.linspace(0, 1, steps=shape[-3], device=device)
        grids = torch.stack(torch.meshgrid([zz, yy, xx]), dim=0)
        grids = grids.unsqueeze(0).repeat(shape[0], 1, 1, 1, 1)
        grids = grids.flatten(1, 2)

    return grids


def compute_locations(h, w, stride, device):
     shifts_x = torch.arange(
         0, w , step=stride,
         dtype=torch.float32, device=device
     )
     shifts_y = torch.arange(
         0, h, step=stride,
         dtype=torch.float32, device=device
     )
     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
     shift_x = shift_x.reshape(-1) + stride // 2
     shift_y = shift_y.reshape(-1) + stride // 2
     shift_x = shift_x.clamp(0, w - 1).long()
     shift_y = shift_y.clamp(0, h - 1).long()
     return shift_x, shift_y


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def named_apply(fn, module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
