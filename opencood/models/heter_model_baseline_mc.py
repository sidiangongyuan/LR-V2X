# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, V2XViTFusion, CoBEVT
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.utils.packet_loss_utils import build_spatial_packet_loss_mask
import importlib
import torchvision

class HeterModelBaselineMC(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineMC, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "fcooper":
            self.fusion_net = SpatialFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])

        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * args['num_class'] * args['num_class'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'] * args['num_class'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'] * args['num_class'],
                                  kernel_size=1) # BIN_NUM = 2
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()


        self.compression_ratio = args.get('compression_ratio', 1.0)
        self.packet_loss_mode = args.get('packet_loss_mode', 'bernoulli')
        self.burst_coarse_h = int(args.get('burst_coarse_h', 8))
        self.burst_coarse_w = int(args.get('burst_coarse_w', 16))
        self.temporal_block_len = int(args.get('temporal_block_len', 1))
        self.packet_loss_seed_base = args.get('packet_loss_seed_base', None)
        self._apply_packet_loss_in_encoder = True

        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def _encode_bev_features(self, data_dict, output_dict):
        agent_modality_list = data_dict['agent_modality_list']
        record_len = data_dict['record_len']
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self._apply_packet_loss_in_encoder and self.compression_ratio < 1.0:
            _, _, height, width = heter_feature_2d.shape
            mask = build_spatial_packet_loss_mask(
                record_len=record_len,
                spatial_size=(height, width),
                keep_ratio=self.compression_ratio,
                device=heter_feature_2d.device,
                dtype=heter_feature_2d.dtype,
                mode=self.packet_loss_mode,
                burst_coarse_shape=(self.burst_coarse_h, self.burst_coarse_w),
                temporal_block_len=self.temporal_block_len,
                sample_indices=data_dict.get('sample_idx'),
                seed_base=self.packet_loss_seed_base,
            )
            heter_feature_2d = heter_feature_2d * mask

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        self._bev_features = heter_feature_2d
        return heter_feature_2d

    def forward(self, data_dict):
        output_dict = {}
        affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size
        )
        record_len = data_dict['record_len']
        heter_feature_2d = self._encode_bev_features(data_dict, output_dict)

        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        # fused_feature = self.fusion_net(heter_feature_2d, record_len)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'fused_feature': fused_feature})
        
        output_dict.update({'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1)})

        return output_dict

    def get_memory_footprint(self):
            """Calculate the total memory footprint of the model's parameters and buffers."""
            total_size = 0
            for param in self.parameters():
                total_size += param.nelement() * param.element_size()
            for buffer in self.buffers():
                total_size += buffer.nelement() * buffer.element_size()

            total_size_MB = total_size / (1024 ** 2)  # Convert to MB
            return f"Model Memory Footprint: {total_size_MB:.2f} MB"
