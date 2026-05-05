"""
Latent Encoder for BEV Feature Compression

Compresses high-dimensional BEV features to compact spatial latent representations
for efficient V2X transmission.

Input: [B, 64, 100, 352] BEV features
Output: [B, 16, 13, 44] compressed latent (~35KB per agent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEncoder(nn.Module):
    """
    Progressive BEV feature compression to compact spatial latent.

    Architecture:
        Stage 1: Conv(64->32) + Downsample(H->H/2, W->W/2)
        Stage 2: Conv(32->16) + Downsample(H/2->H/4, W/2->W/4)
        Stage 3: AdaptiveAvgPool2d to target spatial size

    Design principles:
        1. Preserve spatial structure (not global vector like codebook)
        2. Progressive downsampling for gradual information compression
        3. GroupNorm for stable training
        4. SiLU activation for smooth gradients
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_dim: int = 16,
        target_spatial: tuple = (13, 44),
    ):
        """
        Args:
            in_channels: Input BEV feature channels (default: 64)
            latent_dim: Output latent dimension (default: 16)
            target_spatial: Target spatial size (H, W) after compression
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.target_spatial = target_spatial

        # Stage 1: 64 -> 32 channels, spatial /2
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )

        # Stage 2: 32 -> 16 channels, spatial /2
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, latent_dim),
            nn.SiLU(inplace=True),
        )

        # Stage 3: Adaptive pooling to target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress BEV features to compact latent.

        Args:
            x: [B, in_channels, H, W] BEV features
               Typical: [B, 64, 100, 352]

        Returns:
            z: [B, latent_dim, target_H, target_W] compressed latent
               Typical: [B, 16, 13, 44]
        """
        # Stage 1: [B, 64, 100, 352] -> [B, 32, 50, 176]
        h = self.stage1(x)

        # Stage 2: [B, 32, 50, 176] -> [B, 16, 25, 88]
        h = self.stage2(h)

        # Stage 3: [B, 16, 25, 88] -> [B, 16, 13, 44]
        z = self.adaptive_pool(h)

        return z

    def get_compression_ratio(self):
        """
        Calculate compression ratio.

        Returns:
            ratio: compression ratio (input_size / output_size)
        """
        # Assuming input [64, 100, 352] and output [16, 13, 44]
        input_size = self.in_channels * 100 * 352  # 2,252,800
        output_size = self.latent_dim * self.target_spatial[0] * self.target_spatial[1]  # 9,152
        ratio = input_size / output_size
        return ratio

    def get_transmission_size_kb(self):
        """
        Calculate transmission size in KB (assuming float32).

        Returns:
            size_kb: transmission size per agent in KB
        """
        num_elements = self.latent_dim * self.target_spatial[0] * self.target_spatial[1]
        bytes_per_element = 4  # float32
        size_kb = (num_elements * bytes_per_element) / 1024
        return size_kb


if __name__ == "__main__":
    # Test the encoder
    encoder = LatentEncoder(
        in_channels=64,
        latent_dim=16,
        target_spatial=(13, 44)
    )

    # Test input
    x = torch.randn(2, 64, 100, 352)

    # Forward pass
    z = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Compression ratio: {encoder.get_compression_ratio():.2f}x")
    print(f"Transmission size: {encoder.get_transmission_size_kb():.2f} KB per agent")
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
