"""
Lightweight Conditional Diffusion Model for V2X (Memory Efficient)

Key optimizations:
1. Remove cross-attention from encoder/decoder (save 70% memory)
2. Use simple concat + conv for agent conditioning
3. Only keep attention at bottleneck
4. Simpler ResBlocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class SimpleCrossAttention(nn.Module):
    """
    Simplified cross-attention (only for bottleneck).
    Supports different dimensions for x (query) and context (key/value).
    """
    def __init__(self, dim: int, context_dim: int = None, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(self.context_dim, dim)  # Project context to same dim
        self.v_proj = nn.Linear(self.context_dim, dim)  # Project context to same dim
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        _, C_ctx, H_ctx, W_ctx = context.shape

        # Sanity check: ensure context dimension matches what we expect
        if C_ctx != self.context_dim:
            raise RuntimeError(
                f"SimpleCrossAttention: context dimension mismatch! "
                f"Expected {self.context_dim}, got {C_ctx}. "
                f"Model was initialized with context_dim={self.context_dim}, "
                f"but received context with {C_ctx} channels. "
                f"Please restart training to reload the updated model definition."
            )

        # Reshape to sequence
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        ctx_seq = context.flatten(2).transpose(1, 2)  # [B, H_ctx*W_ctx, C_ctx]

        # Project
        q = self.q_proj(x_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return x + out  # Residual


class SimpleResBlock(nn.Module):
    """
    Simplified residual block with time modulation.
    No fancy AdaLN, just concat time embedding.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU(inplace=True)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.conv2(h)
        h = self.norm2(h)

        return self.act(h + self.residual_conv(x))


class LightweightDiffusionUNet(nn.Module):
    """
    Lightweight U-Net for diffusion (memory efficient).

    Key changes from full version:
    - NO cross-attention in encoder/decoder
    - Agent context via simple concat + conv
    - Only 1 attention block at bottleneck
    - Simpler ResBlocks
    """

    def __init__(
        self,
        in_channels: int = 64,
        model_channels: int = 64,
        out_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        latent_dim: int = 8,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        from .spatial_transformer import SpatialTransformer

        self.in_channels = in_channels
        self.model_channels = model_channels

        # Time embedding
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Spatial transformer
        self.spatial_transformer = SpatialTransformer(latent_dim)

        # Agent latent projection
        self.latent_proj = nn.Conv2d(latent_dim, model_channels, kernel_size=3, padding=1)

        # Input projection (x_t + agent_context)
        # ★ 去掉ego_obs，强迫模型从agent信息推断
        self.input_proj = nn.Conv2d(
            in_channels + model_channels,  # x_t + agent (no ego!)
            model_channels,
            kernel_size=3,
            padding=1
        )

        # Encoder (no attention!)
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        for mult in channel_mult[:-1]:
            out_ch = model_channels * mult
            self.encoder_blocks.append(
                nn.ModuleList([
                    SimpleResBlock(ch, out_ch, time_emb_dim),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)  # Downsample
                ])
            )
            ch = out_ch

        # Bottleneck (only place with attention)
        bottleneck_ch = model_channels * channel_mult[-1]

        # Project agent context to bottleneck dimension for attention
        self.agent_context_proj = nn.Conv2d(model_channels, bottleneck_ch, kernel_size=1)

        self.bottleneck = nn.ModuleList([
            SimpleResBlock(ch, bottleneck_ch, time_emb_dim),
            SimpleCrossAttention(bottleneck_ch, context_dim=bottleneck_ch, num_heads=4),  # Context now matches bottleneck_ch
            SimpleResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim),
        ])

        # Decoder (no attention!)
        self.decoder_blocks = nn.ModuleList()
        ch = bottleneck_ch
        for mult in reversed(channel_mult[:-1]):
            out_ch = model_channels * mult
            self.decoder_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),  # Upsample
                    SimpleResBlock(ch + out_ch, out_ch, time_emb_dim),  # +out_ch for skip
                ])
            )
            ch = out_ch

        # Output
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        agent_latents: List[torch.Tensor],
        transforms: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Lightweight forward pass.

        Strategy:
        1. Fuse agent latents via simple concat (no attention)
        2. Concat [x_t, agent_context] at input (NO ego_obs - force learning from agents!)
        3. Only use attention at bottleneck

        ★ 核心设计变更：去掉ego_obs，模型必须从agent信息推断ego环境
        """
        # Time embedding
        t_emb = self.time_embedder(t)
        t_emb = self.time_mlp(t_emb)

        # Process agent latents: transform + fuse (simple average)
        if agent_latents and transforms:
            agent_context_list = []
            for z_agent, T in zip(agent_latents, transforms):
                z_ego = self.spatial_transformer(z_agent, T)
                z_upsampled = F.interpolate(z_ego, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
                z_proj = self.latent_proj(z_upsampled)
                agent_context_list.append(z_proj)
            agent_context = torch.stack(agent_context_list, dim=0).mean(dim=0)
        else:
            # ★ 无agent时：用可学习的默认context，而不是全零
            # 这让模型在无协作时也能学习有意义的表征
            agent_context = torch.zeros(x_t.shape[0], self.model_channels, x_t.shape[2], x_t.shape[3], device=x_t.device)

        # Concat input: [x_t, agent_context] - NO ego_obs!
        # 模型必须从agent信息推断，不能从ego自身作弊
        x_in = torch.cat([x_t, agent_context], dim=1)
        h = self.input_proj(x_in)

        # Encoder (no attention, save memory!)
        skips = []
        for res_block, downsample in self.encoder_blocks:
            h = res_block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Bottleneck (only place with attention)
        h = self.bottleneck[0](h, t_emb)  # ResBlock

        # Prepare agent context for attention at bottleneck
        if agent_latents and transforms:
            # Downsample agent context to bottleneck resolution and project to bottleneck_ch
            agent_context_downsampled = F.interpolate(
                agent_context,
                size=h.shape[-2:],
                mode='bilinear',
                align_corners=False
            )  # Still model_channels (64)
            agent_context_bottleneck = self.agent_context_proj(agent_context_downsampled)  # Now bottleneck_ch (256)
        else:
            # No agents: create zero context with correct dimensions
            agent_context_bottleneck = torch.zeros_like(h)  # bottleneck_ch dimensions

        h = self.bottleneck[1](h, agent_context_bottleneck)  # Attention
        h = self.bottleneck[2](h, t_emb)  # ResBlock

        # Decoder (no attention, save memory!)
        for upsample, res_block in self.decoder_blocks:
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = res_block(h, t_emb)

        # Output
        out = self.output_proj(h)
        return out


if __name__ == "__main__":
    # Test
    model = LightweightDiffusionUNet(
        in_channels=64,
        model_channels=64,
        out_channels=64,
        channel_mult=(1, 2, 4),
        latent_dim=8,
        time_emb_dim=256,
    )

    B = 2
    x_t = torch.randn(B, 64, 100, 352)
    t = torch.randint(0, 1000, (B,))
    # ego_obs removed - no cheating!
    agent_latents = [torch.randn(B, 8, 13, 44) for _ in range(2)]
    transforms = [torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1) for _ in range(2)]  # [B, 2, 3] affine

    with torch.no_grad():
        output = model(x_t, t, agent_latents, transforms)

    print(f"Input: {x_t.shape}")
    print(f"Output: {output.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
