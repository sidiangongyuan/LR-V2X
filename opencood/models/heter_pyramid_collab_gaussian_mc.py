"""
Heterogeneous Pyramid Collaboration with Gaussian Splatting.

Based on HeterPyramidCollab, replacing dense feature transmission with Gaussian representation.

Author: Claude
"""

import torch
import torch.nn as nn
from collections import Counter
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_mc import HeterPyramidCollabMC
from opencood.models.sub_modules.gaussian_modules import (
    GaussianProposalNetwork,
    GaussianSplatting,
    GaussianFusionModule,
    gaussian_communication_cost
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm
import torchvision


class HeterPyramidCollabGaussianMC(HeterPyramidCollabMC):
    """
    Gaussian Splatting-based cooperative perception model.
    """
    def __init__(self, args):
        super(HeterPyramidCollabGaussianMC, self).__init__(args)

        # Gaussian configuration
        gaussian_config = args.get('gaussian', {})
        self.num_gaussians = gaussian_config.get('num_gaussians', 500)
        self.feature_dim = gaussian_config.get('feature_dim', 64)
        self.use_spatial_attention = gaussian_config.get('use_spatial_attention', False)

        # Get actual output channels from backbone config for each modality
        self.aligner_output_channels = {}
        for modality_name in self.modality_name_list:
            backbone_config = args[modality_name]['backbone_args']
            num_filters = backbone_config['num_filters']
            self.aligner_output_channels[modality_name] = num_filters[-1]  # last layer output

        # Gaussian Proposal Networks (one per modality)
        for modality_name in self.modality_name_list:
            in_channels = self.aligner_output_channels[modality_name]
            setattr(
                self,
                f"gaussian_proposal_{modality_name}",
                GaussianProposalNetwork(
                    in_channels=in_channels,
                    feature_dim=self.feature_dim,
                    num_gaussians=self.num_gaussians
                )
            )

        # Gaussian Fusion Module
        self.gaussian_fusion = GaussianFusionModule(
            feature_dim=self.feature_dim,
            use_spatial_attention=self.use_spatial_attention
        )

        # Gaussian Splatting (render back to dense BEV)
        # Use the first modality's output channels as default
        default_output_channels = list(self.aligner_output_channels.values())[0]
        self.gaussian_splatting = GaussianSplatting(
            feature_dim=self.feature_dim,
            output_channels=default_output_channels
        )

        # Flag to enable/disable gaussian mode (for ablation)
        self.use_gaussian = gaussian_config.get('use_gaussian', True)

        # Auxiliary loss weights
        self.lambda_recon = gaussian_config.get('lambda_recon', 0.1)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}

        # Handle agent_modality_list (might be tensor codes or string list)
        raw_mod_list = data_dict['agent_modality_list']
        if isinstance(raw_mod_list, torch.Tensor):
            idxs = raw_mod_list.tolist()
            agent_modality_list = [self.modality_name_list[i-1] for i in idxs]
        else:
            agent_modality_list = raw_mod_list

        # Get transformation matrices
        pairwise_t_matrix_original = data_dict['pairwise_t_matrix'].to(torch.float32)  # [B, L, L, 4, 4]
        affine_matrix = normalize_pairwise_tfm(pairwise_t_matrix_original, self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # 1. Encode each modality
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # 2. Crop/Pad camera feature map (if needed)
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        # 3. Assemble heterogeneous features
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        agent_modalities_used = []

        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            agent_modalities_used.append(modality_name)
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)  # [sum(record_len), C, H, W]

        # Get spatial size from actual features
        _, _, H_feat, W_feat = heter_feature_2d.shape
        spatial_size = (H_feat, W_feat)


        # if not self.training:
        #     print("=== Original Dense Feature ===")
        #     print(f"Dense - min: {heter_feature_2d.min():.6f}, "
        #             f"max: {heter_feature_2d.max():.6f}, "
        #             f"mean: {heter_feature_2d.mean():.6f}")
      
        # === Gaussian Splatting Logic ===
        if self.use_gaussian:
            # Store original dense features for reconstruction loss
            dense_features_original = heter_feature_2d.clone()

            # 4. Convert to Gaussians (per agent) for communication
            all_gaussians = []
            for idx, (feat, modality) in enumerate(zip(heter_feature_2d, agent_modalities_used)):
                feat_single = feat.unsqueeze(0)  # [1, C, H, W]
                gaussian_proposal_module = eval(f"self.gaussian_proposal_{modality}")
                gaussians = gaussian_proposal_module(feat_single, return_importance=False)
                all_gaussians.append(gaussians)

            # Stack all gaussians for communication cost calculation
            gaussians_all = {
                'xyz': torch.cat([g['xyz'] for g in all_gaussians], dim=0),
                'features': torch.cat([g['features'] for g in all_gaussians], dim=0),
                'scales': torch.cat([g['scales'] for g in all_gaussians], dim=0),
                'opacities': torch.cat([g['opacities'] for g in all_gaussians], dim=0)
            }
            # Calculate communication cost (for logging)

            # if not self.training:
            #     print("=== Gaussian Debug Info ===")
            #     print(f"Opacities - min: {gaussians_all['opacities'].min():.6f}, "
            #             f"max: {gaussians_all['opacities'].max():.6f}, "
            #             f"mean: {gaussians_all['opacities'].mean():.6f}")
            #     print(f"Scales - min: {gaussians_all['scales'].min():.6f}, "
            #             f"max: {gaussians_all['scales'].max():.6f}")
            #     print(f"Rendered features - min: {heter_feature_2d.min():.6f}, "
            #             f"max: {heter_feature_2d.max():.6f}, "
            #             f"mean: {heter_feature_2d.mean():.6f}")
            #     print(f"Has NaN: {torch.isnan(heter_feature_2d).any()}")
            #     print(f"Has Inf: {torch.isinf(heter_feature_2d).any()}")


            comm_cost = gaussian_communication_cost(gaussians_all, num_bytes_per_param=4)
            output_dict['comm_cost_kb'] = comm_cost

            # === This is where communication happens ===
            # In real deployment, only gaussians (not dense features) would be transmitted
            # Transmission size: N * (3 + D + 2 + 1) floats per agent

            # 5. Render each agent's Gaussians back to dense (NO fusion here)
            # Fusion will be done by pyramid_backbone later (using traditional warp method)
            rendered_features = []
            for gaussians in all_gaussians:
                rendered = self.gaussian_splatting(gaussians, spatial_size=spatial_size)
                rendered_features.append(rendered.squeeze(0))  # [C, H, W]

            # Stack to [sum(record_len), C, H, W] - same shape as before Gaussian
            heter_feature_2d = torch.stack(rendered_features, dim=0)

            # 6. Compute reconstruction loss (auxiliary)
            if self.training and self.lambda_recon > 0:
                reconstruction_loss = nn.functional.mse_loss(heter_feature_2d, dense_features_original)
                output_dict['reconstruction_loss'] = reconstruction_loss * self.lambda_recon
            else:
                output_dict['reconstruction_loss'] = torch.tensor(0.0, device=heter_feature_2d.device)

        # === End Gaussian Logic ===

        # Downsample if needed (compressor)
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # 8. Fusion module (Pyramid or other)
        fused_feature, occ_outputs = self.pyramid_backbone(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info
        )

        # 9. Detection heads
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds
        })

        output_dict.update({'occ_single_list': occ_outputs})
        output_dict.update({'preds_tensor': torch.cat([cls_preds, reg_preds, dir_preds], dim=1)})

        return output_dict
