"""
Conditional Diffusion Model for V2X Feature Generation
Implements a U-Net based diffusion model with cross-attention for multi-agent fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time step embedding for diffusion models.
    Converts scalar timestep t to a high-dimensional vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] tensor of timesteps
        Returns:
            [B, dim] sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning on agent features.
    Query: ego feature, Key/Value: agent context
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] query (ego feature)
            context: [B, C, H, W] key/value (agent context)
        Returns:
            [B, C, H, W] attended feature
        """
        B, C, H, W = x.shape

        # Reshape to sequence format
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        ctx_seq = context.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Project and reshape for multi-head attention
        q = self.q_proj(x_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [B, num_heads, H*W, head_dim]
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate and project
        attn_out = attn_weights @ v  # [B, num_heads, H*W, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, C)  # [B, H*W, C]
        attn_out = self.out_proj(attn_out)

        # Residual connection and reshape back
        out = self.norm(x_seq + attn_out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization with scale and shift modulation.
    Used to inject time and condition information.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input feature
            cond: [B, cond_dim] condition vector (e.g., time embedding)
        Returns:
            [B, C, H, W] modulated feature
        """
        B, C, H, W = x.shape

        # Normalize spatial features
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_seq)

        # Get scale and shift from condition
        scale_shift = self.modulation(cond)  # [B, 2*C]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, C], [B, C]

        # Apply modulation
        out = x_norm * (1 + scale[:, None, :]) + shift[:, None, :]
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class ResidualBlock(nn.Module):
    """
    Residual block with time modulation via AdaLN.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.adaln = AdaLNModulation(in_channels, time_emb_dim)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU(inplace=True)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] input feature
            t_emb: [B, time_emb_dim] time embedding
        Returns:
            [B, C_out, H, W] output feature
        """
        # Apply time modulation
        h = self.adaln(x, t_emb)

        # First conv block
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual connection
        out = self.act(h + self.residual_conv(x))

        return out


class DownBlock(nn.Module):
    """
    Downsampling block: ResBlock -> CrossAttention -> Downsample
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 has_cross_attn: bool = True, num_heads: int = 8):
        super().__init__()

        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)

        if has_cross_attn:
            self.cross_attn = CrossAttentionBlock(out_channels, num_heads=num_heads)
        else:
            self.cross_attn = None

        # Downsample
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: [B, C_in, H, W]
            t_emb: [B, time_emb_dim]
            context: [B, C_out, H, W] agent context (optional)
        Returns:
            (downsampled, skip): downsampled feature and skip connection
        """
        h = self.res_block(x, t_emb)

        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        skip = h
        h = self.downsample(h)

        return h, skip


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Concat(skip) -> ResBlock -> CrossAttention
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 has_cross_attn: bool = True, num_heads: int = 8):
        super().__init__()

        # Upsample
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

        # ResBlock with concatenated skip connection
        self.res_block = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)

        if has_cross_attn:
            self.cross_attn = CrossAttentionBlock(out_channels, num_heads=num_heads)
        else:
            self.cross_attn = None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
            skip: [B, C_out, H*2, W*2] skip connection from encoder
            t_emb: [B, time_emb_dim]
            context: [B, C_out, H*2, W*2] agent context (optional)
        Returns:
            [B, C_out, H*2, W*2] upsampled feature
        """
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        h = self.res_block(h, t_emb)

        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        return h


class ConditionalDiffusionUNet(nn.Module):
    """
    Conditional U-Net for diffusion-based V2X feature generation.

    Architecture:
        - Input: noisy BEV [B, 64, 100, 352] + ego observation
        - Conditioning: agent latents [B, 16, 13, 44] transformed to ego frame
        - Output: denoised BEV [B, 64, 100, 352]

    Features:
        - Time embedding via sinusoidal + MLP
        - Cross-attention at multiple resolutions for multi-agent fusion
        - AdaLN for time and condition injection
        - U-Net with skip connections
    """
    def __init__(
        self,
        in_channels: int = 64,  # BEV feature channels
        model_channels: int = 128,  # Base model channels
        out_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple = (1, 2, 4, 8),  # Channel multipliers for each level
        num_heads: int = 8,
        latent_dim: int = 16,  # Agent latent dimension
        time_emb_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        from .spatial_transformer import SpatialTransformer

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Spatial transformer for agent latents
        self.spatial_transformer = SpatialTransformer(latent_dim)

        # Agent latent projection
        self.latent_proj = nn.Sequential(
            nn.Conv2d(latent_dim, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(inplace=True),
        )

        # Input projection (concat noisy x_t with ego observation)
        self.input_proj = nn.Conv2d(in_channels * 2, model_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult[:-1]):  # Don't downsample on last level
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    DownBlock(ch, out_ch, time_emb_dim, has_cross_attn=True, num_heads=num_heads)
                )
                ch = out_ch

        # Bottleneck
        bottleneck_ch = model_channels * channel_mult[-1]
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch, bottleneck_ch, time_emb_dim, dropout),
            CrossAttentionBlock(bottleneck_ch, num_heads=num_heads),
            ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, dropout),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        ch = bottleneck_ch
        channel_mult_reversed = list(reversed(channel_mult[:-1]))
        for i, mult in enumerate(channel_mult_reversed):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.decoder_blocks.append(
                    UpBlock(ch, out_ch, time_emb_dim, has_cross_attn=True, num_heads=num_heads)
                )
                ch = out_ch

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 64, 100, 352] noisy BEV
        t: torch.Tensor,  # [B] timestep
        agent_latents: List[torch.Tensor],  # List of [B, 16, 13, 44] agent latents
        ego_obs: torch.Tensor,  # [B, 64, 100, 352] ego observation (condition)
        transforms: List[torch.Tensor],  # List of [B, 4, 4] transformation matrices
    ) -> torch.Tensor:
        """
        Forward pass of conditional diffusion model.

        Args:
            x_t: Noisy ego BEV feature at timestep t
            t: Diffusion timestep
            agent_latents: List of compressed agent features
            ego_obs: Ego's own observation as condition
            transforms: Transformation matrices from agent to ego frame

        Returns:
            Predicted noise or denoised feature (depending on parameterization)
        """
        # Time embedding
        t_emb = self.time_embedder(t)  # [B, time_emb_dim]
        t_emb = self.time_mlp(t_emb)  # [B, time_emb_dim]

        # Concatenate noisy x_t with ego observation
        x = torch.cat([x_t, ego_obs], dim=1)  # [B, 128, 100, 352]
        h = self.input_proj(x)  # [B, model_channels, 100, 352]

        # Process agent conditions: transform and fuse
        if agent_latents and transforms:
            agent_context_list = []
            for z_agent, T in zip(agent_latents, transforms):
                # Transform to ego coordinate frame
                z_ego = self.spatial_transformer(z_agent, T)  # [B, 16, 13, 44]

                # Upsample to match current feature resolution
                z_upsampled = F.interpolate(
                    z_ego,
                    size=h.shape[-2:],  # Match spatial dimensions
                    mode='bilinear',
                    align_corners=False
                )

                # Project to model dimension
                z_proj = self.latent_proj(z_upsampled)  # [B, model_channels, H, W]
                agent_context_list.append(z_proj)

            # Fuse all agent contexts (simple average, can use learnable fusion)
            agent_context = torch.stack(agent_context_list, dim=0).mean(dim=0)  # [B, model_channels, H, W]
        else:
            # No agent information (ego-only mode)
            agent_context = torch.zeros_like(h)

        # Encoder with skip connections
        skips = []
        for block in self.encoder_blocks:
            # Downsample agent context to match current resolution
            if agent_context.shape[-2:] != h.shape[-2:]:
                context_downsampled = F.interpolate(
                    agent_context,
                    size=h.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                context_downsampled = agent_context

            h, skip = block(h, t_emb, context_downsampled)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck[0](h, t_emb)  # ResBlock

        # Downsample agent context for bottleneck
        if agent_context.shape[-2:] != h.shape[-2:]:
            context_bottleneck = F.interpolate(
                agent_context,
                size=h.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            context_bottleneck = agent_context

        h = self.bottleneck[1](h, context_bottleneck)  # CrossAttention
        h = self.bottleneck[2](h, t_emb)  # ResBlock

        # Decoder with skip connections
        for block in self.decoder_blocks:
            skip = skips.pop()

            # Upsample agent context to match decoder resolution
            if agent_context.shape[-2:] != skip.shape[-2:]:
                context_upsampled = F.interpolate(
                    agent_context,
                    size=skip.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                context_upsampled = agent_context

            h = block(h, skip, t_emb, context_upsampled)

        # Output projection
        out = self.output_proj(h)  # [B, 64, 100, 352]

        return out


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalDiffusionUNet(
        in_channels=64,
        model_channels=128,
        out_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        latent_dim=16,
    ).to(device)

    # Test inputs
    B = 2
    x_t = torch.randn(B, 64, 100, 352).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    ego_obs = torch.randn(B, 64, 100, 352).to(device)

    # Simulate 3 agents
    agent_latents = [torch.randn(B, 16, 13, 44).to(device) for _ in range(3)]
    transforms = [torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device) for _ in range(3)]

    # Forward pass
    with torch.no_grad():
        output = model(x_t, t, agent_latents, ego_obs, transforms)

    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
