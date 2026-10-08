# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.voxel_postprocessor_3heads import VoxelPostprocessor3Heads

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'VoxelPostprocessor3Heads': VoxelPostprocessor3Heads,
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
