import torch
import torch.nn as nn

import torch.nn.functional as F


class SpatialSqueeze(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSqueeze, self).__init__()
        
        self.block = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                   nn.Conv2d(in_channels, in_channels // 2, 
                                             kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels // 2, in_channels, 
                                             kernel_size=1),
                                   nn.Sigmoid())
        
    def forward(self, x):
        return x * self.block(x)

    
class ChannelSqueeze(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSqueeze, self).__init__()
        
        self.block = nn.Sequential(nn.Conv2d(in_channels, 1, 
                                             kernel_size=1),
                                    nn.Sigmoid())
    
    def forward(self, x):
        return x * self.block(x)
        
        
 class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(UpSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        if skip is not None:
            x += skip
        
        return self.block(x)
