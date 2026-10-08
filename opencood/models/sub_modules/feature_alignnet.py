# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from torch import nn


class AlignNet(nn.Module):
    """Identity alignment for the released homogeneous LiDAR configurations."""

    def __init__(self, args: dict):
        super().__init__()
        if args['core_method'] != 'identity' or args.get('spatial_align', False):
            raise ValueError('The released LiDAR configurations use identity alignment.')
        self.channel_align = nn.Identity()

    def forward(self, x):
        return self.channel_align(x)
