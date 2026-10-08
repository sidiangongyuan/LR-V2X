"""
DiffV2X with Pyramid Fusion (Multi-Class Detection)

Integrates:
- Pyramid fusion from heter_pyramid_collab_mc
- Diffusion module (DiT) for feature reconstruction
- Multi-class detection heads

Training stages:
- Stage 2: Train diffusion only (freeze pyramid fusion)
- Stage 3: Fine-tune reconstruction, fusion, and heads with a frozen sensor backbone
"""

import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, Counter
import importlib
import torchvision

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.latent_encoder import LatentEncoder
from opencood.models.sub_modules.diffusion_model_dit import (
    DiT_V2X, DiT_V2X_S, DiT_V2X_B, DiT_V2X_L, DiT_V2X_XL
)
from opencood.models.sub_modules.diffusion_sampler import (
    DiffusionSchedule, DDPMSampler, DDIMSampler
)
from opencood.models.sub_modules.simple_prior_decoder import SimplePriorDecoder
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.packet_loss_utils import build_spatial_packet_loss_mask
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class DiffV2XPyramidMC(nn.Module):
    """
    DiffV2X with Pyramid Fusion for Multi-Class Detection

    Architecture:
        1. Encoder + Backbone + Aligner (from pyramid fusion)
        2. Pyramid fusion for multi-scale collaborative perception
        3. Diffusion module for feature reconstruction
        4. Multi-class detection heads
    """

    def __init__(self, args):
        super(DiffV2XPyramidMC, self).__init__()
        self.args = args

        # Modality setup
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list

        self.num_class = args["num_class"]
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {}

        # Training stage control
        self.use_diffusion = args.get('use_diffusion', True)
        self.diffusion_training_only = args.get('diffusion_training_only', False)
        self.use_prior_only = args.get('use_prior_only', False)  # Ablation: use prior without diffusion

        print(f"\n{'='*80}")
        print(f"DiffV2X Pyramid MC Initialization")
        print(f"{'='*80}")
        print(f"Number of classes: {self.num_class}")
        print(f"Use diffusion: {self.use_diffusion}")
        print(f"Use prior only: {self.use_prior_only}")
        print(f"Diffusion training only: {self.diffusion_training_only}")
        print(f"{'='*80}\n")

        # ============================================================
        # 1. Encoder + Backbone + Aligner (per modality)
        # ============================================================
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # Import encoder
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # Encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"depth_supervision_{modality_name}",
                   model_setting['encoder_args'].get("depth_supervision", False))

            # Backbone
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            # Aligner
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            # Camera crop info
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}",
                       (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}",
                       (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        # For feature transformation
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # ============================================================
        # 2. Pyramid Fusion Backbone
        # ============================================================
        fusion_args = args['fusion_backbone']
        if fusion_args.get("proj_first", False):
            from opencood.models.fuse_modules.pyramid_fuse_onnx import PyramidFusion
        else:
            from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion

        self.pyramid_backbone = PyramidFusion(fusion_args)

        # ============================================================
        # 3. Diffusion Module (if enabled)
        # ============================================================
        if self.use_diffusion:
            diffusion_args = args.get('diffusion', {})
            self.latent_dim = diffusion_args.get('latent_dim', 32)
            self.bev_channels = diffusion_args.get('bev_channels', 64)

            # Latent encoder
            target_spatial = diffusion_args.get('target_spatial', [13, 44])
            self.latent_h, self.latent_w = target_spatial
            self.latent_encoder = LatentEncoder(
                in_channels=self.bev_channels,
                latent_dim=self.latent_dim,
                target_spatial=target_spatial
            )

            # Diffusion schedule
            self.num_timesteps = diffusion_args.get('num_timesteps', 1000)
            self.schedule = DiffusionSchedule(
                num_timesteps=self.num_timesteps,
                beta_schedule=diffusion_args.get('beta_schedule', 'cosine'),
                beta_start=diffusion_args.get('beta_start', 1e-4),
                beta_end=diffusion_args.get('beta_end', 0.02),
            )

            # DiT model
            use_mode = diffusion_args.get('use_mode', 'dit-b')
            dit_config = diffusion_args.get('dit_config', {})

            patch_size = dit_config.get('patch_size', 4)
            spatial_h = dit_config.get('spatial_h', 100)
            spatial_w = dit_config.get('spatial_w', 352)
            num_cross_attn_layers = dit_config.get('num_cross_attn_layers', 9)
            use_ckpt = diffusion_args.get('use_checkpoint', False)

            if use_mode == 'dit-b':
                self.diffusion_model = DiT_V2X_B(
                    in_channels=self.bev_channels,
                    out_channels=self.bev_channels,
                    spatial_size=(spatial_h, spatial_w),
                    patch_size=patch_size,
                    latent_dim=self.latent_dim,
                    num_cross_attn_layers=num_cross_attn_layers,
                    use_checkpoint=use_ckpt,
                )
            elif use_mode == 'dit-s':
                self.diffusion_model = DiT_V2X_S(
                    in_channels=self.bev_channels,
                    out_channels=self.bev_channels,
                    spatial_size=(spatial_h, spatial_w),
                    patch_size=patch_size,
                    latent_dim=self.latent_dim,
                    num_cross_attn_layers=num_cross_attn_layers,
                    use_checkpoint=use_ckpt,
                )
            elif use_mode == 'dit-l':
                self.diffusion_model = DiT_V2X_L(
                    in_channels=self.bev_channels,
                    out_channels=self.bev_channels,
                    spatial_size=(spatial_h, spatial_w),
                    patch_size=patch_size,
                    latent_dim=self.latent_dim,
                    num_cross_attn_layers=num_cross_attn_layers,
                    use_checkpoint=use_ckpt,
                )
            else:
                # Custom DiT
                self.diffusion_model = DiT_V2X(
                    in_channels=self.bev_channels,
                    out_channels=self.bev_channels,
                    spatial_size=(spatial_h, spatial_w),
                    patch_size=patch_size,
                    hidden_size=dit_config.get('hidden_size', 768),
                    depth=dit_config.get('depth', 18),
                    num_heads=dit_config.get('num_heads', 12),
                    mlp_ratio=dit_config.get('mlp_ratio', 4.0),
                    latent_dim=self.latent_dim,
                    num_cross_attn_layers=num_cross_attn_layers,
                    use_checkpoint=use_ckpt,
                )

            # Samplers
            self.sampler_type = diffusion_args.get('sampler_type', 'ddim')
            self.num_inference_steps = diffusion_args.get('num_inference_steps', 20)

            # Prior configuration (zero, statistical, or latent-based)
            self.use_zero_prior = diffusion_args.get('use_zero_prior', False)
            self.use_statistical_prior = diffusion_args.get('use_statistical_prior', False)
            self.use_latent_prior = diffusion_args.get('use_latent_prior', True)
            self.inference_timestep = diffusion_args.get('inference_timestep', 500)

            # Ego condition configuration (NEW)
            self.use_ego_condition = diffusion_args.get('use_ego_condition', True)
            print(f"[DiffV2X] Use ego condition: {self.use_ego_condition}")

            # Load statistical prior if enabled
            if self.use_statistical_prior:
                # Auto-detect prior path from stage1 model directory
                prior_path = diffusion_args.get('prior_path', None)

                # If prior_path not specified, try to auto-detect from args
                if prior_path is None:
                    stage1_dir = args.get('stage1_model_dir', None)
                    if stage1_dir and os.path.isdir(stage1_dir):
                        prior_path = os.path.join(stage1_dir, 'bev_prior_statistics.pth')
                        print(f"[DiffV2X] Auto-detected prior path: {prior_path}")

                if prior_path and os.path.exists(prior_path):
                    print(f"[DiffV2X] Loading BEV prior from: {prior_path}")
                    prior_stats = torch.load(prior_path, map_location='cpu')
                    # Register as buffer (not trainable, but saved in checkpoint)
                    self.register_buffer('bev_mean', prior_stats['bev_mean'])
                    self.register_buffer('bev_std', prior_stats['bev_std'])
                    print(f"[DiffV2X] BEV prior loaded successfully:")
                    print(f"  - Shape: {self.bev_mean.shape}")
                    print(f"  - Mean range: [{self.bev_mean.min():.4f}, {self.bev_mean.max():.4f}]")
                    print(f"  - Std range: [{self.bev_std.min():.4f}, {self.bev_std.max():.4f}]")
                    print(f"  - Computed from {prior_stats['num_samples']} samples")
                elif prior_path:
                    # Prior file doesn't exist yet - need to compute
                    print(f"[DiffV2X] WARNING: BEV prior not found at: {prior_path}")
                    print(f"[DiffV2X] Please compute BEV prior first using:")
                    print(f"  python opencood/tools/compute_bev_prior.py \\")
                    print(f"    --model_dir <stage1_model_dir> \\")
                    print(f"    --output_path {prior_path}")
                    print(f"[DiffV2X] Falling back to zero prior for now")
                    self.use_statistical_prior = False
                    self.register_buffer('bev_mean', None)
                    self.register_buffer('bev_std', None)
                else:
                    print(f"[DiffV2X] WARNING: Cannot determine prior path")
                    print(f"[DiffV2X] Please specify either 'prior_path' or 'stage1_model_dir' in config")
                    print(f"[DiffV2X] Falling back to zero prior")
                    self.use_statistical_prior = False
                    self.register_buffer('bev_mean', None)
                    self.register_buffer('bev_std', None)
            else:
                self.register_buffer('bev_mean', None)
                self.register_buffer('bev_std', None)

            # Initialize latent prior decoder if enabled
            if self.use_latent_prior:
                print(f"[DiffV2X] Initializing Latent Prior Decoder")
                # Use self.bev_channels (already defined from diffusion config)
                # This is the correct BEV feature channel dimension
                self.prior_decoder = SimplePriorDecoder(
                    latent_dim=self.latent_dim,
                    bev_channels=self.bev_channels,  # Use self.bev_channels (64)
                    latent_spatial=(self.latent_h, self.latent_w),
                    bev_spatial=(spatial_h, spatial_w)  # Feature map size [100, 352]
                )
                num_params = self.prior_decoder.get_num_parameters()
                print(f"[DiffV2X] Latent Prior Decoder initialized:")
                print(f"  - Input: [{self.latent_dim}, {self.latent_h}, {self.latent_w}]")
                print(f"  - Output: [{self.bev_channels}, {spatial_h}, {spatial_w}]")
                print(f"  - Parameters: {num_params:,}")
            else:
                self.prior_decoder = None

            self.train_sampler = DDPMSampler(self.schedule, loss_type='mse')
            # DDIM sampler with x0-prediction (JiT style)
            self.sampler = DDIMSampler(
                self.schedule,
                eta=diffusion_args.get('ddim_eta', 0.0),
                prediction_type="x0"  # Model directly predicts clean BEV
            )

            self.compression_ratio = diffusion_args.get('compression_ratio', 1.0)
            self.packet_loss_mode = diffusion_args.get(
                'packet_loss_mode', 'bernoulli'
            )
            self.burst_coarse_h = int(diffusion_args.get('burst_coarse_h', 8))
            self.burst_coarse_w = int(diffusion_args.get('burst_coarse_w', 16))
            self.temporal_block_len = int(
                diffusion_args.get('temporal_block_len', 1)
            )
            self.packet_loss_seed_base = diffusion_args.get(
                'packet_loss_seed_base'
            )

            print(f"[DiffV2X] Diffusion module initialized:")
            print(f"  - Model: {use_mode}")
            print(f"  - Latent dim: {self.latent_dim}")
            print(f"  - Prediction type: x0 (direct MSE loss)")
            print(f"  - Use zero prior: {self.use_zero_prior}")
            print(f"  - Use statistical prior: {self.use_statistical_prior}")
            print(f"  - Use latent prior: {self.use_latent_prior}")
            print(f"  - Inference timestep: {self.inference_timestep}")
            print(f"  - Compression ratio: {self.compression_ratio}")

        # ============================================================
        # 4. Shrink Header (optional)
        # ============================================================
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # ============================================================
        # 5. Detection Heads
        # ============================================================
        self.cls_head = nn.Conv2d(args['in_head'],
                                  args['anchor_number'] * args['num_class'] * args['num_class'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'],
                                  7 * args['anchor_number'] * args['num_class'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'],
                                  args['dir_args']['num_bins'] * args['anchor_number'] * args['num_class'],
                                  kernel_size=1)

        check_trainable_module(self)


    def get_memory_footprint(self):
            """Calculate the total memory footprint of the model's parameters and buffers."""
            total_size = 0
            for param in self.parameters():
                total_size += param.nelement() * param.element_size()
            for buffer in self.buffers():
                total_size += buffer.nelement() * buffer.element_size()

            total_size_MB = total_size / (1024 ** 2)  # Convert to MB
            return f"Model Memory Footprint: {total_size_MB:.2f} MB"
    
    
    @staticmethod
    def _normalize_modality_indicator(x):
        """Normalize modality indicator to string like 'm1'."""
        if isinstance(x, torch.Tensor):
            return f"m{int(x.item())}"
        elif isinstance(x, str):
            return x
        elif isinstance(x, (int, np.integer)):
            return f"m{int(x)}"
        else:
            raise TypeError(f"Unexpected type for modality: {type(x)}")

    def _normalize_modality_list(self, agent_modality_list):
        return [self._normalize_modality_indicator(x) for x in agent_modality_list]

    def forward(self, data_dict, enable_vis=False):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']

        # Visualization data collection
        if enable_vis and not self.training:
            output_dict['vis_data'] = {
                'priors': [],
                'reconstructed': [],
                'ground_truth': []
            }
        affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'],
            self.H, self.W, self.fake_voxel_size
        ).to(torch.float32)
        record_len = data_dict['record_len']

        # ============================================================
        # 1. Extract features per modality
        # ============================================================
        modality_list_normalized = self._normalize_modality_list(agent_modality_list)
        modality_count_dict = Counter(modality_list_normalized)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(feature)
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # Crop camera features if needed
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)

        # Assemble heterogeneous features
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for x in agent_modality_list:
            modality_name = self._normalize_modality_indicator(x)
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)  # [N_agents, C, H, W]

        # ============================================================
        # 2. Diffusion-based feature reconstruction (if enabled)
        # ============================================================
        if self.use_diffusion:
            # Compress to latents
            agent_latents = self.latent_encoder(heter_feature_2d)  # [N_agents, latent_dim, h, w]

            # DEBUG: Save original features
            self._original_bev_features = heter_feature_2d.detach().clone()

            # ============================================================
            # ============================================================
            # x0-PREDICTION with Velocity Loss (JiT style)
            # Following "Back to Basics: Let Denoising Generative Models Denoise"
            # Network directly predicts clean BEV (x0), but loss is computed in velocity space
            # ============================================================

            # Apply compression (simulate communication constraint)
            if self.compression_ratio < 1.0:
                _, _, H, W = agent_latents.shape
                mask = build_spatial_packet_loss_mask(
                    record_len=record_len,
                    spatial_size=(H, W),
                    keep_ratio=self.compression_ratio,
                    device=agent_latents.device,
                    dtype=agent_latents.dtype,
                    mode=self.packet_loss_mode,
                    burst_coarse_shape=(self.burst_coarse_h, self.burst_coarse_w),
                    temporal_block_len=self.temporal_block_len,
                    sample_indices=data_dict.get('sample_idx'),
                    seed_base=self.packet_loss_seed_base,
                )
                agent_latents_compressed = agent_latents * mask
            else:
                agent_latents_compressed = agent_latents

            # Prepare for diffusion training
            batch_size = len(record_len)
            diffusion_losses = []
            reconstructed_features = []

            start_idx = 0
            for b_idx in range(batch_size):
                num_agents = record_len[b_idx]
                end_idx = start_idx + num_agents

                agent_feats_b = heter_feature_2d[start_idx:end_idx]
                agent_latents_b = agent_latents_compressed[start_idx:end_idx]

                # Get ego BEV for this batch (first agent is ego)
                ego_bev_b = agent_feats_b[0:1]  # [1, C, H, W]
                _, C_bev, H_bev, W_bev = ego_bev_b.shape

                # Get transformation matrices for this batch
                # affine_matrix: [B, L, L, 2, 3]
                # affine_matrix[b, i, j] is transform from agent_j to agent_i
                T_matrices_b = affine_matrix[b_idx, :num_agents, :num_agents, :, :]  # [N, N, 2, 3]

                for agent_idx in range(num_agents):
                    x0 = agent_feats_b[agent_idx:agent_idx+1]  # Clean BEV (ground truth)
                    agent_latent = agent_latents_b[agent_idx:agent_idx+1]

                    # Skip ego (no reconstruction needed)
                    if agent_idx == 0:
                        # Ego: use original features
                        reconstructed_features.append(x0.detach() if self.training else x0)
                        continue

                    # Other agents: use diffusion to reconstruct
                    # Transform ego BEV to agent coordinate frame
                    if self.use_ego_condition:
                        # T_matrices_b[agent_idx, 0] is transform: ego -> agent
                        # We need to warp ego BEV to agent's coordinate frame
                        T_ego_to_agent = T_matrices_b[agent_idx, 0:1, :, :]  # [1, 2, 3]

                        # Warp ego BEV to agent coordinate frame
                        ego_bev_in_agent_coord = warp_affine_simple(
                            ego_bev_b,  # [1, C, H, W]
                            T_ego_to_agent,  # [1, 2, 3]
                            (H_bev, W_bev)
                        )  # [1, C, H, W]
                    else:
                        ego_bev_in_agent_coord = None
                        T_ego_to_agent = None

                    if self.training:
                        # ===== Training: x0-prediction with prior =====
                        # Sample random timestep (avoid t too close to num_timesteps for stability)
                        max_t = int(0.999 * self.num_timesteps)
                        t = torch.randint(0, max_t, (1,), device=x0.device).long()

                        # Generate noisy input x_t
                        noise = torch.randn_like(x0)

                        # Determine prior based on configuration
                        if self.use_latent_prior:
                            # Latent-based prior: decode latent to rough BEV with target size
                            x0_prior = self.prior_decoder(agent_latent, target_size=x0.shape[2:])
                        elif self.use_statistical_prior:
                            # Statistical prior: use BEV mean
                            x0_prior = self.bev_mean.unsqueeze(0).to(x0.device)
                        elif self.use_zero_prior:
                            # Zero-prior mode: x_t = q_sample(0, t, noise)
                            x0_prior = torch.zeros_like(x0)
                        else:
                            # Standard mode: use ground truth x0
                            x0_prior = x0

                        x_t = self.train_sampler.q_sample(x0_prior, t, noise)

                        # Network predicts clean x0 (NOT noise!)
                        predicted_x0 = self.diffusion_model(
                            x_t,  # Noisy input
                            t,    # Timestep
                            agent_latents=[agent_latent],
                            transforms=[T_ego_to_agent] if T_ego_to_agent is not None else [],
                            ego_bev=ego_bev_in_agent_coord  # NEW: ego condition
                        )

                        # Direct x0-prediction loss (simple MSE)
                        # Loss is always against true x0, regardless of prior
                        loss = nn.functional.mse_loss(predicted_x0, x0)
                        diffusion_losses.append(loss)

                        # For fusion during training: use predicted_x0 directly
                        # Detach to avoid double backprop through fusion path
                        reconstructed_features.append(predicted_x0.detach())

                    else:
                        # ===== Validation/Inference =====
                        # Determine prior (must match training)
                        if self.use_latent_prior:
                            # Latent-based prior: decode latent to rough BEV with target size
                            x0_prior = self.prior_decoder(agent_latent, target_size=x0.shape[2:])
                        elif self.use_statistical_prior:
                            # Statistical prior: use BEV mean (no transmission needed!)
                            x0_prior = self.bev_mean.unsqueeze(0).to(x0.device)
                        elif self.use_zero_prior:
                            # Zero-prior mode: matches zero-prior training
                            x0_prior = torch.zeros_like(x0)
                        else:
                            # Standard mode: use ground truth x0 (for validation only)
                            # NOTE: This cannot be used in real V2X deployment
                            x0_prior = x0

                        # ABLATION: Use prior only (skip diffusion refinement)
                        if self.use_prior_only:
                            # Use prior directly without diffusion
                            predicted_x0 = x0_prior
                            loss = nn.functional.mse_loss(predicted_x0, x0)
                            diffusion_losses.append(loss)
                        else:
                            # Standard diffusion refinement
                            # Fixed timestep for deterministic inference
                            t_inference = torch.full((1,), self.inference_timestep, device=x0.device, dtype=torch.long)

                            # Generate x_t (method depends on prior configuration)
                            noise = torch.randn_like(x0)  # Deterministic: seed fixed in inference_mc.py
                            x_t = self.train_sampler.q_sample(x0_prior, t_inference, noise)

                            # Single-step prediction
                            predicted_x0 = self.diffusion_model(
                                x_t,
                                t_inference,
                                agent_latents=[agent_latent],
                                transforms=[T_ego_to_agent] if T_ego_to_agent is not None else [],
                                ego_bev=ego_bev_in_agent_coord  # NEW: ego condition
                            )

                            # Validation loss (MSE on x0)
                            # We still have x0 for validation to monitor reconstruction quality
                            loss = nn.functional.mse_loss(predicted_x0, x0)
                            diffusion_losses.append(loss)

                        # Use predicted x0 for fusion
                        reconstructed_features.append(predicted_x0)

                        # Save visualization data if enabled
                        if enable_vis:
                            output_dict['vis_data']['priors'].append(x0_prior.detach().cpu())
                            output_dict['vis_data']['reconstructed'].append(predicted_x0.detach().cpu())
                            output_dict['vis_data']['ground_truth'].append(x0.detach().cpu())

                            # Compute objectness heatmaps for better visualization
                            # Use L2 norm across channels as importance measure (no trained weights needed)
                            with torch.no_grad():
                                # Prior heatmap: L2 norm across channels
                                prior_heatmap = torch.norm(x0_prior, p=2, dim=1, keepdim=True)  # [1, 1, H, W]
                                prior_heatmap = prior_heatmap / (prior_heatmap.max() + 1e-8)  # Normalize to [0, 1]

                                # Reconstructed heatmap
                                recon_heatmap = torch.norm(0.5*x0+0.5*predicted_x0, p=2, dim=1, keepdim=True)
                                recon_heatmap = recon_heatmap / (recon_heatmap.max() + 1e-8)

                                # Ground truth heatmap
                                gt_heatmap = torch.norm(x0, p=2, dim=1, keepdim=True)
                                gt_heatmap = gt_heatmap / (gt_heatmap.max() + 1e-8)

                                # Save heatmaps
                                if 'prior_heatmaps' not in output_dict['vis_data']:
                                    output_dict['vis_data']['prior_heatmaps'] = []
                                    output_dict['vis_data']['recon_heatmaps'] = []
                                    output_dict['vis_data']['gt_heatmaps'] = []

                                output_dict['vis_data']['prior_heatmaps'].append(prior_heatmap.detach().cpu())
                                output_dict['vis_data']['recon_heatmaps'].append(recon_heatmap.detach().cpu())
                                output_dict['vis_data']['gt_heatmaps'].append(gt_heatmap.detach().cpu())

                start_idx = end_idx

            # Compute diffusion loss (ensure it requires grad)
            if diffusion_losses:
                output_dict['diffusion_loss'] = torch.stack(diffusion_losses).mean()
            else:
                # If no agents need reconstruction (rare case: all samples are ego-only)
                # Create a dummy loss from model parameters to maintain gradient graph
                dummy_loss = sum(p.sum() for p in self.diffusion_model.parameters() if p.requires_grad) * 0.0
                output_dict['diffusion_loss'] = dummy_loss

            # Concatenate all reconstructed features
            features_for_fusion = torch.cat(reconstructed_features, dim=0)

            # CRITICAL: Ensure non-negative features (BEV features should be non-negative after ReLU)
            # This prevents distribution mismatch between training and inference
            features_for_fusion = torch.relu(features_for_fusion)

            # DEBUG: Save reconstructed features
            self._reconstructed_features = features_for_fusion.detach().clone()

        else:
            # No diffusion: use original features
            features_for_fusion = heter_feature_2d
            output_dict['diffusion_loss'] = torch.tensor(0.0, device=heter_feature_2d.device)

        # ============================================================
        # 3. Pyramid Fusion
        # ============================================================
        fused_feature, occ_outputs = self.pyramid_backbone(
            features_for_fusion,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info
        )

        # ============================================================
        # 4. Detection Heads
        # ============================================================
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            # For feature distribution visualization (e.g., t-SNE under packet loss).
            # This is the fused feature map fed into the detection heads (after optional shrink).
            'fused_feature': fused_feature,
            'occ_single_list': occ_outputs
        })

        return output_dict
