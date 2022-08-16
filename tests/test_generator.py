from fanjiang.simulation import physics
import numpy as np
import torch
from fanjiang.gan import MSGenerator


def compute_intervals(coords, resolution=6371 * 1000 * np.pi / 180.0):
    latitude, longitude = coords
    grid_x = longitude * resolution
    grid_y = latitude * resolution
    dx_m = grid_x[:, 1:] - grid_x[:, :-1]

    dx_l = torch.cat([dx_m[:, [0]], dx_m], dim=1)
    dx_r = torch.cat([dx_m, dx_m[:, [-1]]], dim=1)    
    dx = torch.stack([dx_l, dx_r])

    dy_m = grid_y[:, 1:] - grid_y[:, :-1]
    dy_l = torch.cat([dy_m[:, [0]], dy_m], dim=1)
    dy_r = torch.cat([dy_m, dy_m[:, [-1]]], dim=1)    
    dy = torch.stack([dy_l, dy_r])
    return (dx, dy)

def test():
    fields = torch.rand(1, 40, 60, 60)
    net = MSGenerator(
        in_channels=40, 
        out_channels=40, 
        hidden_channels=[(128, 256, 256, 128), (160, 160, 160, 160)], 
        physic_channels=0,
    )

    latitude = torch.arange(0, 15, 0.25).view(1, -1).repeat(1, 1)
    longitude = torch.arange(0, 15, 0.25).view(1, -1).repeat(1, 1)
    # print(latitude.shape)
    grids = compute_intervals([latitude, longitude])

    print(net)

    outputs = net(grids, fields, 7)
    for y in outputs:
        print(y.shape)

test()