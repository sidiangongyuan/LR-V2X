"""
Simple Prior Decoder for Latent-based Prior Generation
Lightweight decoder to convert compressed latent back to rough BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePriorDecoder(nn.Module):
    """
    Lightweight decoder to generate rough x0_prior from agent latent.
    Used as a better prior than zeros for diffusion reconstruction.

    Args:
        latent_dim: Channel dimension of latent features
        bev_channels: Channel dimension of output BEV features
        latent_spatial: (H, W) spatial size of latent
        bev_spatial: (H, W) spatial size of BEV
    """
    def __init__(
        self,
        latent_dim=64,
        bev_channels=256,
        latent_spatial=(26, 88),
        bev_spatial=(100, 352)
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.bev_channels = bev_channels
        self.latent_h, self.latent_w = int(latent_spatial[0]), int(latent_spatial[1])
        self.bev_h, self.bev_w = int(bev_spatial[0]), int(bev_spatial[1])

        # Calculate upsampling ratios
        self.h_ratio = self.bev_h / self.latent_h
        self.w_ratio = self.bev_w / self.latent_w

        # Simple 3-layer ConvTranspose decoder
        # Designed to be lightweight (~1M parameters)

        mid_channels = 128

        self.decoder = nn.Sequential(
            # Stage 1: [64, 26, 88] → [128, 52, 176]
            nn.ConvTranspose2d(
                latent_dim, mid_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Stage 2: [128, 52, 176] → [256, 104, 352]
            nn.ConvTranspose2d(
                mid_channels, bev_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        # Additional refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent, target_size=None):
        """
        Args:
            latent: [B, latent_dim, latent_h, latent_w]
            target_size: Optional (H, W) tuple for dynamic sizing.
                         If None, uses self.bev_h and self.bev_w

        Returns:
            rough_bev: [B, bev_channels, bev_h, bev_w]
        """
        # Decode
        rough_bev = self.decoder(latent)  # [B, bev_channels, ~bev_h, ~bev_w]

        # Determine target size
        if target_size is not None:
            target_h, target_w = int(target_size[0]), int(target_size[1])
        else:
            target_h, target_w = self.bev_h, self.bev_w

        # Resize to target size using interpolation (more flexible than crop/pad)
        B, C, H, W = rough_bev.shape
        if H != target_h or W != target_w:
            rough_bev = F.interpolate(rough_bev, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)

        # Refinement
        rough_bev = self.refine(rough_bev)

        return rough_bev

    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test
    decoder = SimplePriorDecoder(
        latent_dim=64,
        bev_channels=256,
        latent_spatial=(26, 88),
        bev_spatial=(100, 352)
    )

    print(f"Prior Decoder Parameters: {decoder.get_num_parameters():,}")

    # Test forward
    latent = torch.randn(2, 64, 26, 88)
    rough_bev = decoder(latent)

    print(f"Input shape: {latent.shape}")
    print(f"Output shape: {rough_bev.shape}")

    assert rough_bev.shape == (2, 256, 100, 352), "Output shape mismatch!"
    print("Test passed!")
