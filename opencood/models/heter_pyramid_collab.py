""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion 
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.utils.packet_loss_utils import build_spatial_packet_loss_mask
import importlib
import torchvision

class HeterPyramidCollab(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

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
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        
        fusion_args = args['fusion_backbone']
        if fusion_args.get("proj_first", False):
            from opencood.models.fuse_modules.pyramid_fuse_onnx import PyramidFusion
        else:
            from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion

        self.pyramid_backbone = PyramidFusion(fusion_args)


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
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
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

        comm_bottleneck_args = args.get('comm_bottleneck', {})
        self.comm_bottleneck_enabled = bool(comm_bottleneck_args.get('enabled', False))
        self.comm_bottleneck_factor = int(comm_bottleneck_args.get('factor', 8))
        self.comm_bottleneck_downsample_mode = comm_bottleneck_args.get(
            'downsample_mode',
            'area',
        )
        self.comm_bottleneck_upsample_mode = comm_bottleneck_args.get(
            'upsample_mode',
            'bilinear',
        )
        self.comm_bottleneck_apply_to_ego = bool(
            comm_bottleneck_args.get('apply_to_ego', False)
        )
        self.comm_bottleneck_align_corners = comm_bottleneck_args.get(
            'align_corners',
            False,
        )
        if self.comm_bottleneck_factor < 1:
            raise ValueError(
                f"comm_bottleneck.factor must be >= 1, got {self.comm_bottleneck_factor}"
            )

        if self.compress and self.comm_bottleneck_enabled:
            raise ValueError(
                "compressor and comm_bottleneck should not be enabled together; "
                "they represent different communication protocols."
            )

        if self.comm_bottleneck_enabled:
            print(
                "[CommBottleneck] enabled "
                f"(factor={self.comm_bottleneck_factor}, "
                f"down={self.comm_bottleneck_downsample_mode}, "
                f"up={self.comm_bottleneck_upsample_mode}, "
                f"apply_to_ego={self.comm_bottleneck_apply_to_ego})"
            )
        
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def _interpolate_2d(self, x, size, mode):
        if mode in {'linear', 'bilinear', 'bicubic', 'trilinear'}:
            return F.interpolate(
                x,
                size=size,
                mode=mode,
                align_corners=bool(self.comm_bottleneck_align_corners),
            )
        return F.interpolate(x, size=size, mode=mode)

    def _get_comm_bottleneck_size(self, height, width):
        if height % self.comm_bottleneck_factor != 0 or width % self.comm_bottleneck_factor != 0:
            raise ValueError(
                "Communication bottleneck requires spatial sizes divisible by factor, "
                f"got {(height, width)} with factor {self.comm_bottleneck_factor}"
            )
        return height // self.comm_bottleneck_factor, width // self.comm_bottleneck_factor

    def _apply_comm_bottleneck(self, heter_feature_2d, record_len, data_dict):
        _, _, height, width = heter_feature_2d.shape
        comm_h, comm_w = self._get_comm_bottleneck_size(height, width)

        transmitted_feature = self._interpolate_2d(
            heter_feature_2d,
            size=(comm_h, comm_w),
            mode=self.comm_bottleneck_downsample_mode,
        )

        if self.compression_ratio < 1.0:
            comm_mask = build_spatial_packet_loss_mask(
                record_len=record_len,
                spatial_size=(comm_h, comm_w),
                keep_ratio=self.compression_ratio,
                device=transmitted_feature.device,
                dtype=transmitted_feature.dtype,
                mode=self.packet_loss_mode,
                burst_coarse_shape=(self.burst_coarse_h, self.burst_coarse_w),
                temporal_block_len=self.temporal_block_len,
                sample_indices=data_dict.get('sample_idx'),
                seed_base=self.packet_loss_seed_base,
            )
            transmitted_feature = transmitted_feature * comm_mask

        restored_feature = self._interpolate_2d(
            transmitted_feature,
            size=(height, width),
            mode=self.comm_bottleneck_upsample_mode,
        )

        if not self.comm_bottleneck_apply_to_ego:
            ego_mask = build_spatial_packet_loss_mask(
                record_len=record_len,
                spatial_size=(height, width),
                keep_ratio=0.0,
                device=heter_feature_2d.device,
                dtype=heter_feature_2d.dtype,
            )
            restored_feature = restored_feature * (1.0 - ego_mask) + heter_feature_2d * ego_mask

        return restored_feature

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        # --- debug & remap integer codes into your ["m1","m2",...] names ---
        raw_mod_list = data_dict['agent_modality_list']
        if isinstance(raw_mod_list, torch.Tensor):
            idxs = raw_mod_list.tolist()
            # print(">>> agent_modality_list codes:", idxs)
            # turn 1-based codes [1,1,1...] into zero-based indices [0,0,0...]
            agent_modality_list = [
                self.modality_name_list[i-1] for i in idxs
            ]
        else:
            # already a list of strings?
            agent_modality_list = raw_mod_list

        # now agent_modality_list is e.g. ["m1","m1","m1","m1","m1"]
        # print(">>> agent_modality_list (final):", agent_modality_list)

        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name) # output changes to tensor here
            feature = eval(f"self.backbone_{modality_name}")(feature) # input changes to tensor here
            feature = eval(f"self.aligner_{modality_name}")(feature)
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
        # print("agent_modality_list:", agent_modality_list,
        #     type(agent_modality_list), 
        #     [type(x) for x in (agent_modality_list if isinstance(agent_modality_list, (list,tuple)) else [])])
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)


        # Communication protocol:
        # 1) Optional fixed spatial bottleneck for collaborator features.
        # 2) Packet loss is applied on the transmitted representation.
        # 3) Pyramid fusion still receives features at the original spatial resolution.
        if self.comm_bottleneck_enabled:
            heter_feature_2d = self._apply_comm_bottleneck(
                heter_feature_2d,
                record_len,
                data_dict,
            )
        elif self.compression_ratio < 1.0:
            _, _, H, W = heter_feature_2d.shape
            mask = build_spatial_packet_loss_mask(
                record_len=record_len,
                spatial_size=(H, W),
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

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        fused_feature, occ_outputs = self.pyramid_backbone(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )

        if self.shrink_flag:
            shrinked_feature = self.shrink_conv(fused_feature)
            feature_for_head = shrinked_feature
            output_dict.update({'feature_map_after_shrink': shrinked_feature})
        else:
            feature_for_head = fused_feature

        cls_preds = self.cls_head(feature_for_head)
        reg_preds = self.reg_head(feature_for_head)
        dir_preds = self.dir_head(feature_for_head)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1)})

        # output_dict.update({'feat_map_encoder': heter_feature_2d})
        output_dict.update(
            {
                'feature_map_after_fusion': feature_for_head,
                # For feature distribution visualization (e.g., t-SNE under packet loss).
                # This is the fused feature map fed into the detection heads (after optional shrink).
                'fused_feature': feature_for_head,
            }
        )
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

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

    def forward_onnx_export(self,
                            voxel_features,
                            voxel_coords,
                            voxel_num_points,
                            pairwise_t_matrix,
                            record_len,
                            agent_modality_list):
        # Dummy ops to ensure these inputs are used in the computation graph
        pairwise_t_matrix = pairwise_t_matrix + 0 * pairwise_t_matrix
        record_len = record_len + 0 * record_len
        agent_modality_list = agent_modality_list + 0 * agent_modality_list

        data_dict = {
            'pairwise_t_matrix': pairwise_t_matrix,
            'record_len': record_len,
            'agent_modality_list': agent_modality_list,
            'inputs_m1': {
                'voxel_features': voxel_features,
                'voxel_coords':   voxel_coords,
                'voxel_num_points': voxel_num_points,
            }
        }
        return self.forward(data_dict)
