"""
Spatial Transformer for Multi-Agent Feature Alignment

Transforms agent features from their local coordinate frame to ego's coordinate frame
using differentiable grid sampling.

Core challenge: Agent's compressed latent is in agent's coordinate system,
but we need to fuse it in ego's coordinate system for diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Spatial transformation module for agent-to-ego coordinate alignment.

    Key idea:
        1. Create a sampling grid in ego's coordinate system
        2. Apply inverse transformation to get corresponding positions in agent's system
        3. Use grid_sample to sample from agent's latent feature map

    This allows differentiable transformation of spatial features.
    """

    def __init__(self, latent_dim: int = 16):
        """
        Args:
            latent_dim: Dimension of latent features (used for potential learned components)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Note: This is a purely geometric transformation, no learnable parameters needed
        # But we keep the structure for potential future extensions (e.g., learned refinement)

    def forward(
        self,
        z_agent: torch.Tensor,
        T_agent_to_ego: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform agent's latent feature to ego's coordinate frame.

        Args:
            z_agent: [B, C, H, W] agent's latent feature in agent's frame
            T_agent_to_ego: [B, 2, 3] or [2, 3] affine transformation matrix
                           Format: [[a, b, tx], [c, d, ty]]
                           This is the standard PyTorch affine matrix format

        Returns:
            z_ego: [B, C, H, W] agent's feature transformed to ego's frame
        """
        B, C, H, W = z_agent.shape
        device = z_agent.device

        # Handle shape: if input is [2, 3], expand to [B, 2, 3]
        if T_agent_to_ego.dim() == 2:
            T_agent_to_ego = T_agent_to_ego.unsqueeze(0).expand(B, -1, -1)

        # Ensure correct shape [B, 2, 3]
        if T_agent_to_ego.shape != (B, 2, 3):
            raise ValueError(f"Expected T_agent_to_ego shape [{B}, 2, 3], got {T_agent_to_ego.shape}")

        # PyTorch's affine_grid expects the transformation matrix
        # T_agent_to_ego transforms agent->ego, we need ego->agent for sampling
        # Compute inverse of 2x3 affine matrix manually

        a = T_agent_to_ego[:, 0, 0]
        b = T_agent_to_ego[:, 0, 1]
        tx = T_agent_to_ego[:, 0, 2]
        c = T_agent_to_ego[:, 1, 0]
        d = T_agent_to_ego[:, 1, 1]
        ty = T_agent_to_ego[:, 1, 2]

        # Determinant
        det = a * d - b * c
        det = det.clamp(min=1e-6)  # Avoid division by zero

        # Inverse transformation matrix [B, 2, 3]
        T_ego_to_agent = torch.zeros(B, 2, 3, device=device, dtype=T_agent_to_ego.dtype)
        T_ego_to_agent[:, 0, 0] = d / det
        T_ego_to_agent[:, 0, 1] = -b / det
        T_ego_to_agent[:, 0, 2] = (-d * tx + b * ty) / det
        T_ego_to_agent[:, 1, 0] = -c / det
        T_ego_to_agent[:, 1, 1] = a / det
        T_ego_to_agent[:, 1, 2] = (c * tx - a * ty) / det

        # Create sampling grid using inverse transform
        grid_agent = F.affine_grid(
            T_ego_to_agent,
            z_agent.shape,
            align_corners=False
        )  # [B, H, W, 2]

        # Sample from agent's feature map
        z_ego = F.grid_sample(
            z_agent,
            grid_agent,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return z_ego

    def forward_with_physical_coords(
        self,
        z_agent: torch.Tensor,
        T_agent_to_ego: torch.Tensor,
        spatial_range: tuple = (-140.8, -40, 140.8, 40),  # (x_min, y_min, x_max, y_max)
    ) -> torch.Tensor:
        """
        Alternative forward that handles physical coordinates explicitly.

        Args:
            z_agent: [B, C, H, W] agent's latent feature
            T_agent_to_ego: [B, 4, 4] transformation matrix
            spatial_range: (x_min, y_min, x_max, y_max) in meters

        Returns:
            z_ego: [B, C, H, W] transformed feature
        """
        B, C, H, W = z_agent.shape
        device = z_agent.device

        x_min, y_min, x_max, y_max = spatial_range

        # Create grid in physical coordinates
        y_coords = torch.linspace(y_min, y_max, H, device=device)
        x_coords = torch.linspace(x_min, x_max, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Homogeneous coordinates [H, W, 4]
        ones = torch.ones_like(xx)
        zeros = torch.zeros_like(xx)
        grid_ego_phys = torch.stack([xx, yy, zeros, ones], dim=-1)  # [H, W, 4]

        # Expand for batch
        grid_ego_phys = grid_ego_phys.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 4]

        # Flatten
        grid_ego_flat = grid_ego_phys.reshape(B, -1, 4)

        # Transform to agent's coordinate
        T_ego_to_agent = torch.inverse(T_agent_to_ego)
        grid_agent_flat = torch.bmm(grid_ego_flat, T_ego_to_agent.transpose(1, 2))

        # Extract x, y
        grid_agent_x = grid_agent_flat[:, :, 0]
        grid_agent_y = grid_agent_flat[:, :, 1]

        # Normalize to [-1, 1]
        grid_agent_x_norm = 2 * (grid_agent_x - x_min) / (x_max - x_min) - 1
        grid_agent_y_norm = 2 * (grid_agent_y - y_min) / (y_max - y_min) - 1

        # Stack and reshape
        grid_agent_norm = torch.stack([grid_agent_x_norm, grid_agent_y_norm], dim=-1)
        grid_agent_norm = grid_agent_norm.reshape(B, H, W, 2)

        # Sample
        z_ego = F.grid_sample(
            z_agent,
            grid_agent_norm,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return z_ego


if __name__ == "__main__":
    # Test the transformer
    transformer = SpatialTransformer(latent_dim=16)

    # Test input
    B = 2
    z_agent = torch.randn(B, 16, 13, 44)

    # Identity transformation (should return same feature)
    T_identity = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    z_ego_identity = transformer(z_agent, T_identity)

    print(f"Agent feature shape: {z_agent.shape}")
    print(f"Transformed feature shape: {z_ego_identity.shape}")
    print(f"Identity transform preserves features: {torch.allclose(z_agent, z_ego_identity, atol=1e-5)}")

    # Test with actual transformation (translation)
    T_translate = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    T_translate[:, 0, 3] = 10.0  # Translate 10m in x
    z_ego_translated = transformer(z_agent, T_translate)

    print(f"Translated feature shape: {z_ego_translated.shape}")
    print(f"Translation changes features: {not torch.allclose(z_agent, z_ego_translated, atol=1e-5)}")
