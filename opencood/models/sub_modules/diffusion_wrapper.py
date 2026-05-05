"""
Standalone Diffusion Module - Can be combined with any fusion method.

This module is independent of specific fusion implementations and can be
plugged into any collaborative perception model.

Author: Claude
Date: 2025-01-26
"""

import torch
import torch.nn as nn
from opencood.models.sub_modules.latent_encoder import LatentEncoder
from opencood.models.sub_modules.diffusion_model_dit import DiT_V2X, DiT_V2X_B, DiT_V2X_S, DiT_V2X_L
from opencood.models.sub_modules.diffusion_sampler import DiffusionSchedule, DDPMSampler, DDIMSampler


class DiffusionModule(nn.Module):
    """
    Standalone diffusion module for feature reconstruction.

    Can be trained separately and combined with any fusion method.

    Args:
        bev_channels: Number of channels in BEV features (e.g., 64)
        latent_dim: Dimension of compressed latent (e.g., 32)
        spatial_size: Spatial size of BEV features [H, W] (e.g., [100, 352])
        dit_config: Configuration for DiT model
        diffusion_config: Configuration for diffusion process
    """

    def __init__(
        self,
        bev_channels: int = 64,
        latent_dim: int = 32,
        spatial_size: list = [100, 352],
        dit_config: dict = None,
        diffusion_config: dict = None
    ):
        super(DiffusionModule, self).__init__()

        # Default configs
        dit_config = dit_config or {}
        diffusion_config = diffusion_config or {}

        self.bev_channels = bev_channels
        self.latent_dim = latent_dim
        self.spatial_size = tuple(spatial_size)

        # Latent Encoder: BEV features -> compressed latent
        self.latent_encoder = LatentEncoder(
            in_channels=bev_channels,
            latent_dim=latent_dim,
            target_spatial=diffusion_config.get('target_spatial', [13, 44])
        )

        # Diffusion Schedule
        self.num_timesteps = diffusion_config.get('num_timesteps', 1000)
        self.schedule = DiffusionSchedule(
            num_timesteps=self.num_timesteps,
            beta_schedule=diffusion_config.get('beta_schedule', 'cosine'),
            beta_start=diffusion_config.get('beta_start', 1e-4),
            beta_end=diffusion_config.get('beta_end', 0.02),
        )

        # DiT Model
        use_mode = diffusion_config.get('use_mode', 'dit-b')
        patch_size = dit_config.get('patch_size', 4)
        spatial_h = dit_config.get('spatial_h', spatial_size[0])
        spatial_w = dit_config.get('spatial_w', spatial_size[1])
        num_cross_attn_layers = dit_config.get('num_cross_attn_layers', 9)
        use_ckpt = diffusion_config.get('use_checkpoint', False)

        if use_mode == 'dit-b':
            self.diffusion_model = DiT_V2X_B(
                in_channels=bev_channels,
                out_channels=bev_channels,
                spatial_size=(spatial_h, spatial_w),
                patch_size=patch_size,
                latent_dim=latent_dim,
                num_cross_attn_layers=num_cross_attn_layers,
                use_checkpoint=use_ckpt,
            )
        elif use_mode == 'dit-s':
            self.diffusion_model = DiT_V2X_S(
                in_channels=bev_channels,
                out_channels=bev_channels,
                spatial_size=(spatial_h, spatial_w),
                patch_size=patch_size,
                latent_dim=latent_dim,
                num_cross_attn_layers=num_cross_attn_layers,
                use_checkpoint=use_ckpt,
            )
        elif use_mode == 'dit-l':
            self.diffusion_model = DiT_V2X_L(
                in_channels=bev_channels,
                out_channels=bev_channels,
                spatial_size=(spatial_h, spatial_w),
                patch_size=patch_size,
                latent_dim=latent_dim,
                num_cross_attn_layers=num_cross_attn_layers,
                use_checkpoint=use_ckpt,
            )
        else:
            # Custom DiT
            self.diffusion_model = DiT_V2X(
                in_channels=bev_channels,
                out_channels=bev_channels,
                spatial_size=(spatial_h, spatial_w),
                patch_size=patch_size,
                hidden_size=dit_config.get('hidden_size', 768),
                depth=dit_config.get('depth', 18),
                num_heads=dit_config.get('num_heads', 12),
                mlp_ratio=dit_config.get('mlp_ratio', 4.0),
                latent_dim=latent_dim,
                num_cross_attn_layers=num_cross_attn_layers,
                use_checkpoint=use_ckpt,
            )

        # Initialize DiT weights
        self.diffusion_model.initialize_weights()

        # Samplers
        self.train_sampler = DDPMSampler(self.schedule, loss_type='mse')
        self.num_inference_steps = diffusion_config.get('num_inference_steps', 20)
        self.sampler = DDIMSampler(self.schedule, eta=diffusion_config.get('ddim_eta', 0.0))

        self.compression_ratio = diffusion_config.get('compression_ratio', 1.0)

        print(f"[DiffusionModule] Initialized:")
        print(f"  - Model: {use_mode}")
        print(f"  - BEV channels: {bev_channels}")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Spatial size: {spatial_size}")
        print(f"  - Compression ratio: {self.compression_ratio}")

    def compute_diffusion_loss(self, bev_features, record_len):
        """
        Compute diffusion training loss.

        Args:
            bev_features: [N_agents, C, H, W] BEV features from fusion
            record_len: List of agent counts per batch

        Returns:
            diffusion_loss: Scalar loss value
        """
        # Compress to latents
        agent_latents = self.latent_encoder(bev_features)  # [N_agents, latent_dim, h, w]

        batch_size = len(record_len)
        diffusion_losses = []

        start_idx = 0
        for b_idx in range(batch_size):
            num_agents = record_len[b_idx]
            end_idx = start_idx + num_agents

            # Get agent features and latents for this batch
            agent_feats_b = bev_features[start_idx:end_idx]  # [N, C, H, W]
            agent_latents_b = agent_latents[start_idx:end_idx]  # [N, latent_dim, h, w]

            # Reconstruct each agent
            for agent_idx in range(num_agents):
                agent_feat_gt = agent_feats_b[agent_idx:agent_idx+1]  # [1, C, H, W]
                agent_latent = agent_latents_b[agent_idx:agent_idx+1]  # [1, latent_dim, h, w]

                # Simulate compression
                # Ego is local; do not simulate packet loss on ego features.
                if self.compression_ratio < 1.0 and agent_idx != 0:
                    mask = (torch.rand_like(agent_latent) < self.compression_ratio).float()
                    agent_latent_compressed = agent_latent * mask
                else:
                    agent_latent_compressed = agent_latent

                # Diffusion forward process
                t = torch.randint(0, self.num_timesteps, (1,), device=agent_feat_gt.device)
                t = t.expand(agent_feat_gt.shape[0])  # Expand to batch size
                noise = torch.randn_like(agent_feat_gt)
                noisy_feat = self.train_sampler.q_sample(agent_feat_gt, t, noise)

                # Predict noise
                predicted_noise = self.diffusion_model(
                    noisy_feat,
                    t,
                    agent_latents=[agent_latent_compressed],
                    transforms=[]
                )

                # MSE loss
                loss = nn.functional.mse_loss(predicted_noise, noise)
                diffusion_losses.append(loss)

            start_idx = end_idx

        # Average diffusion loss
        if len(diffusion_losses) > 0:
            return torch.stack(diffusion_losses).mean()
        else:
            return torch.tensor(0.0, device=bev_features.device)

    def reconstruct(self, bev_features, record_len=None):
        """
        Reconstruct BEV features using diffusion.

        Args:
            bev_features: [N_agents, C, H, W] BEV features

        Returns:
            reconstructed_features: [N_agents, C, H, W] Reconstructed features
        """
        # Compress to latents
        agent_latents = self.latent_encoder(bev_features)

        # Simulate compression (packet loss) on transmitted features only; keep ego intact.
        if self.compression_ratio < 1.0:
            mask = (torch.rand_like(agent_latents) < self.compression_ratio).float()
            if record_len is None:
                mask[0] = 1.0
            else:
                start_idx = 0
                for num_agents in record_len:
                    num_agents_int = (
                        int(num_agents.item()) if isinstance(num_agents, torch.Tensor) else int(num_agents)
                    )
                    mask[start_idx] = 1.0
                    start_idx += num_agents_int
            agent_latents = agent_latents * mask

        # Reconstruct via DDIM sampling
        reconstructed_features = []
        for agent_idx in range(bev_features.shape[0]):
            agent_latent = agent_latents[agent_idx:agent_idx+1]

            # Get shape for sampling
            shape = bev_features[agent_idx:agent_idx+1].shape

            # DDIM sampling
            x_0 = self.sampler.sample(
                model=self.diffusion_model,
                shape=shape,
                condition={
                    'agent_latents': [agent_latent],
                    'transforms': []
                },
                num_steps=self.num_inference_steps
            )
            reconstructed_features.append(x_0)

        return torch.cat(reconstructed_features, dim=0)

    def forward(self, bev_features, record_len=None, training=True):
        """
        Forward pass.

        Args:
            bev_features: [N_agents, C, H, W] BEV features
            record_len: List of agent counts (only needed for training)
            training: Whether in training mode

        Returns:
            If training: diffusion_loss
            If inference: reconstructed_features
        """
        if training:
            assert record_len is not None, "record_len is required for training"
            return self.compute_diffusion_loss(bev_features, record_len)
        else:
            return self.reconstruct(bev_features, record_len=record_len)
