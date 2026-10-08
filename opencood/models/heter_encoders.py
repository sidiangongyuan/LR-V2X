# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
from torch import nn
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])


    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
    
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict['spatial_features'] # H0, W0
        return lidar_feature_2d
