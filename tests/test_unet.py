import numpy as np
import torch
from fanjiang.cnn import UNet


input_frame = 12
future_frame = 36
radar_channels = 12
sate_channels = 36
field_channels = 0
input_height = input_width = 512

def test_radar_only():
    radars = torch.rand(1, radar_channels, input_height, input_width)
    net = UNet(
        future_frame, 
        radar_channels=radar_channels, 
        field_channels=0, 
        base=2,
        interval=2, 
    )
    y = net([radars, None, None])
    assert y.shape[1:] == (future_frame, input_height, input_width), (y.shape)
    print("radar only pass")


def test_satellite_only():
    satellite = torch.rand(1, sate_channels, input_height//2, input_width//2)
    net = UNet(
        future_frame, 
        sate_channels=sate_channels, 
        sate_layer=1,
        field_channels=0, 
        base=2,
        interval=2, 
    )    
    y = net([None, satellite, None])
    assert y.shape[1:] == (future_frame, input_height, input_width), (y.shape)
    print("satellite only pass")


def test_radar_satellite():
    radars = torch.rand(1, radar_channels, input_height, input_width)
    satellite = torch.rand(1, sate_channels, input_height//4, input_width//4)
    net = UNet(
        future_frame, 
        radar_channels=radar_channels,
        sate_channels=sate_channels, 
        sate_layer=2,
        field_channels=0, 
        base=1.5, 
        interval=2, 
    )    
    y = net([radars, satellite, None])
    assert y.shape[1:] == (future_frame, input_height, input_width), (y.shape)
    print("radar satellite pass")



# test_radar_only()
test_satellite_only()
# test_radar_satellite()