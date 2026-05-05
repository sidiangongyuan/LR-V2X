"""
Gaussian-based modules for cooperative perception.

Author: Claude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianProposalNetwork(nn.Module):
    """
    Convert dense BEV feature to sparse Gaussian representation.

    Args:
        in_channels: input feature channels
        feature_dim: dimension of gaussian feature
        num_gaussians: number of gaussians to generate
    """
    def __init__(self, in_channels=64, feature_dim=64, num_gaussians=500):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_gaussians = num_gaussians

        # Importance prediction branch
        self.importance_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Gaussian feature extraction
        self.feature_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, feature_dim, 1)
        )

        # Gaussian parameters: scale (2), opacity (1)
        self.param_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 1)  # [sigma_x, sigma_y, alpha]
        )

    def forward(self, x, return_importance=False):
        """
        Args:
            x: [B, C, H, W] dense BEV feature

        Returns:
            gaussians: dict containing:
                - xyz: [B, N, 3] positions (x, y, z=0 for BEV)
                - features: [B, N, D] feature vectors
                - scales: [B, N, 2] (sigma_x, sigma_y)
                - opacities: [B, N, 1] alpha values
                - importance_map: [B, 1, H, W] (if return_importance=True)
        """
        B, C, H, W = x.shape

        # 1. Compute importance map
        importance = self.importance_head(x)  # [B, 1, H, W]

        # 2. Extract features for all locations
        features_dense = self.feature_head(x)  # [B, D, H, W]
        params_dense = self.param_head(x)  # [B, 3, H, W]

        # 3. Top-K sampling based on importance
        importance_flat = importance.view(B, -1)  # [B, H*W]
        topk_values, topk_indices = torch.topk(importance_flat, self.num_gaussians, dim=1)  # [B, N]

        # 4. Gather features and params at top-k locations
        # Convert flat indices to 2D coordinates
        topk_y = topk_indices // W  # [B, N]
        topk_x = topk_indices % W   # [B, N]

        # Create position coordinates (actual grid coordinates)
        x_coords = topk_x.float()  # [B, N]
        y_coords = topk_y.float()  # [B, N]
        z_coords = torch.zeros_like(x_coords)  # BEV, z=0
        xyz = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, N, 3]

        # Gather features at top-k locations using proper indexing
        # Flatten spatial dimensions first
        features_flat = features_dense.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # [B, H*W, D]
        params_flat = params_dense.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # [B, H*W, 3]

        # Create batch index
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, self.num_gaussians)  # [B, N]

        # Gather using flat indices
        features = features_flat[batch_idx, topk_indices]  # [B, N, D]
        params = params_flat[batch_idx, topk_indices]  # [B, N, 3]

        # Extract scales and opacities
        scales = torch.clamp(F.softplus(params[..., :2]) + 0.5, min=0.5, max=5.0)  # [B, N, 2], ensure positive, min 0.5
        opacities = torch.sigmoid(params[..., 2:3])  # [B, N, 1], range [0, 1]

        gaussians = {
            'xyz': xyz,
            'features': features,
            'scales': scales,
            'opacities': opacities,
        }

        if return_importance:
            gaussians['importance_map'] = importance

        return gaussians


class GaussianSplatting(nn.Module):
    """
    Render Gaussians back to dense BEV feature map.
    """
    def __init__(self, feature_dim=64, output_channels=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_channels = output_channels

        # Optional: learnable output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, gaussians, spatial_size=None):
        """
        Args:
            gaussians: dict with keys:
                - xyz: [B, N, 3]
                - features: [B, N, D]
                - scales: [B, N, 2]
                - opacities: [B, N, 1]
            spatial_size: (H, W) optional, if not provided, inferred from xyz

        Returns:
            dense_feature: [B, C, H, W]
        """
        B, N, D = gaussians['features'].shape
        device = gaussians['features'].device

        # Infer spatial size from xyz if not provided
        if spatial_size is None:
            xyz = gaussians['xyz']  # [B, N, 3]
            H = int(xyz[..., 1].max().item()) + 5  # y + margin
            W = int(xyz[..., 0].max().item()) + 5  # x + margin
        else:
            H, W = spatial_size

        # Create output grid
        output = torch.zeros(B, D, H, W, device=device)
        normalization = torch.zeros(B, 1, H, W, device=device) + 1e-6

        # Create meshgrid for BEV coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )  # [H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # For each gaussian, compute its contribution
        xyz = gaussians['xyz']  # [B, N, 3]
        features = gaussians['features']  # [B, N, D]
        scales = gaussians['scales']  # [B, N, 2]
        opacities = gaussians['opacities']  # [B, N, 1]

        for i in range(N):
            # Get i-th gaussian parameters
            center_x = xyz[:, i, 0].view(B, 1, 1)  # [B, 1, 1]
            center_y = xyz[:, i, 1].view(B, 1, 1)  # [B, 1, 1]
            sigma_x = scales[:, i, 0].view(B, 1, 1)  # [B, 1, 1]
            sigma_y = scales[:, i, 1].view(B, 1, 1)  # [B, 1, 1]
            alpha = opacities[:, i, 0].view(B, 1, 1)  # [B, 1, 1]
            feat = features[:, i, :].view(B, D, 1, 1)  # [B, D, 1, 1]

            # Compute 2D Gaussian weight
            dx = grid_x - center_x  # [B, H, W]
            dy = grid_y - center_y  # [B, H, W]

            # Gaussian function: exp(-0.5 * ((dx/sigma_x)^2 + (dy/sigma_y)^2))
            exponent = -0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
            weight = alpha * torch.exp(exponent)  # [B, H, W]

            # Accumulate weighted features
            output += feat * weight.unsqueeze(1)  # [B, D, H, W]
            normalization += weight.unsqueeze(1)  # [B, 1, H, W]

        # Normalize
        output = output / normalization

        # Project to output channels
        output = self.output_proj(output)

        return output


class GaussianFusionModule(nn.Module):
    """
    Fuse Gaussians from multiple agents.
    """
    def __init__(self, feature_dim=64, use_spatial_attention=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_spatial_attention = use_spatial_attention

        if use_spatial_attention:
            # Cross-attention between gaussians
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.scale = math.sqrt(feature_dim)

    def forward(self, gaussians_list, record_len, pairwise_t_matrix):
        """
        Args:
            gaussians_list: dict of gaussian from all agents:
                - xyz: [sum(record_len), N, 3]
                - features: [sum(record_len), N, D]
                - scales: [sum(record_len), N, 2]
                - opacities: [sum(record_len), N, 1]
            record_len: list of agent counts per sample
            pairwise_t_matrix: [B, L, L, 4, 4] transformation matrices (ORIGINAL, not normalized)

        Returns:
            fused_gaussians_list: list of B dicts, each containing fused gaussians for one batch:
                - xyz: [1, N_fused, 3] where N_fused = n_agents * N
                - features: [1, N_fused, D]
                - scales: [1, N_fused, 2]
                - opacities: [1, N_fused, 1]
        """
        # Split gaussians by batch
        batch_size = len(record_len)

        # For simplicity, we do simple concatenation after transformation
        all_xyz = gaussians_list['xyz']  # [sum(record_len), N, 3]
        all_features = gaussians_list['features']  # [sum(record_len), N, D]
        all_scales = gaussians_list['scales']  # [sum(record_len), N, 2]
        all_opacities = gaussians_list['opacities']  # [sum(record_len), N, 1]

        fused_gaussians = []

        split_idx = 0
        for b in range(batch_size):
            n_agents = record_len[b]

            # Get gaussians from all agents in this batch
            batch_xyz = all_xyz[split_idx:split_idx + n_agents]  # [n_agents, N, 3]
            batch_features = all_features[split_idx:split_idx + n_agents]  # [n_agents, N, D]
            batch_scales = all_scales[split_idx:split_idx + n_agents]  # [n_agents, N, 2]
            batch_opacities = all_opacities[split_idx:split_idx + n_agents]  # [n_agents, N, 1]

            # Get transformation matrices for this batch
            batch_transforms = pairwise_t_matrix[b, :n_agents, :n_agents, :, :]  # [n_agents, n_agents, 4, 4]

            # Transform all gaussians to ego coordinate (agent 0)
            # We need transforms from each agent to ego: T[0, i] transforms from agent i to ego
            ego_transforms = batch_transforms[0, :, :, :]  # [n_agents, 4, 4]

            # Apply transformation to xyz
            # xyz: [n_agents, N, 3] -> [n_agents, N, 4] (homogeneous)
            xyz_homo = torch.cat([batch_xyz, torch.ones_like(batch_xyz[..., :1])], dim=-1)  # [n_agents, N, 4]

            # Transform: for each agent i, apply ego_transforms[i]
            # ego_transforms[i]: [4, 4]
            # xyz_homo[i]: [N, 4]
            # result: [N, 4]
            xyz_transformed = torch.zeros_like(batch_xyz)  # [n_agents, N, 3]
            for i in range(n_agents):
                # xyz_homo[i]: [N, 4], ego_transforms[i]: [4, 4]
                # result: [N, 4]
                transformed = torch.matmul(xyz_homo[i], ego_transforms[i].transpose(0, 1))  # [N, 4]
                xyz_transformed[i] = transformed[:, :3]  # [N, 3]

            # Simple fusion: concatenate all gaussians
            # [n_agents*N, 3] -> [1, n_agents*N, 3]
            fused_xyz = xyz_transformed.reshape(1, -1, 3)
            fused_features = batch_features.reshape(1, -1, self.feature_dim)
            fused_scales = batch_scales.reshape(1, -1, 2)
            fused_opacities = batch_opacities.reshape(1, -1, 1)

            fused_gaussians.append({
                'xyz': fused_xyz,
                'features': fused_features,
                'scales': fused_scales,
                'opacities': fused_opacities
            })

            split_idx += n_agents

        # Return list of fused gaussians (one per batch)
        # Cannot stack because different batches may have different number of gaussians
        return fused_gaussians


def gaussian_communication_cost(gaussians, num_bytes_per_param=4):
    """
    Calculate communication cost of transmitting gaussians.

    Args:
        gaussians: dict with gaussian parameters
        num_bytes_per_param: bytes per float (4 for fp32, 2 for fp16)

    Returns:
        cost_kb: communication cost in KB
    """
    N = gaussians['xyz'].shape[1]
    D = gaussians['features'].shape[2]

    # xyz: 3, features: D, scales: 2, opacities: 1
    total_params = N * (3 + D + 2 + 1)
    cost_bytes = total_params * num_bytes_per_param
    cost_kb = cost_bytes / 1024

    return cost_kb
