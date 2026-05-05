"""
Hybrid Conditional Diffusion Model for V2X (Memory-Performance Balanced)

Strategy:
- Remove attention from encoder (save memory)
- Keep attention in decoder (preserve performance for multi-agent fusion)
- Only 2-3 attention blocks total (vs 6-8 in full model)

This is a sweet spot between lightweight (12GB) and full model (40GB).
Expected memory: ~20-25GB with batch_size=4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.utils.checkpoint import checkpoint

# Import from lite version
from .diffusion_model_lite import (
    SinusoidalPositionEmbeddings,
    SimpleCrossAttention,
    SimpleResBlock,
)


class HybridDiffusionUNet(nn.Module):
    """
    Hybrid U-Net: No attention in encoder, selective attention in decoder.

    Memory vs Performance tradeoff:
    - Lightweight: 1 attention (bottleneck) -> 12GB, lower performance
    - Hybrid: 3 attentions (bottleneck + 2 decoder) -> 20-25GB, good performance
    - Full: 6-8 attentions (all layers) -> 40GB, best performance
    """

    def __init__(
        self,
        in_channels: int = 64,
        model_channels: int = 64,
        out_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        latent_dim: int = 8,
        time_emb_dim: int = 256,
        num_decoder_attentions: int = 2,  # How many decoder layers have attention
        use_checkpoint: bool = False,  # Gradient checkpointing to save memory
    ):
        super().__init__()

        from .spatial_transformer import SpatialTransformer

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_decoder_attentions = num_decoder_attentions
        self.use_checkpoint = use_checkpoint

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

        # Input projection (x_t + agent_context, NO ego_obs)
        # ★ 去掉ego_obs，强迫模型从agent信息推断
        self.input_proj = nn.Conv2d(
            in_channels + model_channels,  # x_t + agent (no ego!)
            model_channels,
            kernel_size=3,
            padding=1
        )

        # Encoder (NO attention to save memory)
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        for mult in channel_mult[:-1]:
            out_ch = model_channels * mult
            self.encoder_blocks.append(
                nn.ModuleList([
                    SimpleResBlock(ch, out_ch, time_emb_dim),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
                ])
            )
            ch = out_ch

        # Bottleneck (with attention)
        bottleneck_ch = model_channels * channel_mult[-1]
        self.agent_context_proj_bottleneck = nn.Conv2d(model_channels, bottleneck_ch, kernel_size=1)

        self.bottleneck = nn.ModuleList([
            SimpleResBlock(ch, bottleneck_ch, time_emb_dim),
            SimpleCrossAttention(bottleneck_ch, context_dim=bottleneck_ch, num_heads=4),
            SimpleResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim),
        ])

        # Decoder (selective attention - only on first N layers)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_context_projs = nn.ModuleList()
        ch = bottleneck_ch

        for i, mult in enumerate(reversed(channel_mult[:-1])):
            out_ch = model_channels * mult

            # Add attention to first num_decoder_attentions layers
            has_attn = (i < num_decoder_attentions)

            self.decoder_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),
                    SimpleResBlock(ch + out_ch, out_ch, time_emb_dim),
                    SimpleCrossAttention(out_ch, context_dim=out_ch, num_heads=4) if has_attn else None,
                ])
            )

            # Projection for agent context at this decoder level
            if has_attn:
                self.decoder_context_projs.append(nn.Conv2d(model_channels, out_ch, kernel_size=1))
            else:
                self.decoder_context_projs.append(None)

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
        Forward pass: Each agent reconstructs its own features from compressed latent.

        Design: Single-agent reconstruction (no multi-agent mean)
        - Input: Single agent's compressed latent
        - Output: Reconstructed complete features for that agent
        - Training: Agent_i reconstructs itself from z_i
        - Inference: Reconstruct each agent separately, then fuse externally
        """

        # Time embedding
        t_emb = self.time_embedder(t)
        t_emb = self.time_mlp(t_emb)

        # Process single agent latent
        # Design: agent_latents and transforms should contain exactly 1 element
        # representing the agent being reconstructed
        if agent_latents and transforms:
            assert len(agent_latents) == 1, f"Expected single agent, got {len(agent_latents)}"
            assert len(transforms) == 1, f"Expected single transform, got {len(transforms)}"

            z_agent = agent_latents[0]  # [B, latent_dim, h, w]
            T = transforms[0]  # [B, 2, 3]

            # Transform latent to target coordinate frame
            z_transformed = self.spatial_transformer(z_agent, T)
            # Upsample to match x_t spatial size
            z_upsampled = F.interpolate(z_transformed, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
            # Project to model channels
            agent_context = self.latent_proj(z_upsampled)
        else:
            # No agent latent provided (shouldn't happen in normal use)
            agent_context = torch.zeros(x_t.shape[0], self.model_channels, x_t.shape[2], x_t.shape[3], device=x_t.device)

        # Input: [x_t, agent_context] - NO ego_obs!
        x_in = torch.cat([x_t, agent_context], dim=1)
        h = self.input_proj(x_in)

        # Encoder (no attention)
        skips = []
        for res_block, downsample in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint(res_block, h, t_emb, use_reentrant=False)
            else:
                h = res_block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Bottleneck (with attention)
        if self.use_checkpoint and self.training:
            h = checkpoint(self.bottleneck[0], h, t_emb, use_reentrant=False)
        else:
            h = self.bottleneck[0](h, t_emb)

        agent_context_bottleneck_ds = F.interpolate(agent_context, size=h.shape[-2:], mode='bilinear', align_corners=False)
        agent_context_bottleneck = self.agent_context_proj_bottleneck(agent_context_bottleneck_ds)

        # ★ CRITICAL: Use checkpoint for bottleneck attention too
        if self.use_checkpoint and self.training:
            h = checkpoint(self.bottleneck[1], h, agent_context_bottleneck, use_reentrant=False)
        else:
            h = self.bottleneck[1](h, agent_context_bottleneck)

        if self.use_checkpoint and self.training:
            h = checkpoint(self.bottleneck[2], h, t_emb, use_reentrant=False)
        else:
            h = self.bottleneck[2](h, t_emb)

        # Decoder (selective attention)
        for i, (upsample, res_block, cross_attn) in enumerate(self.decoder_blocks):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            if self.use_checkpoint and self.training:
                h = checkpoint(res_block, h, t_emb, use_reentrant=False)
            else:
                h = res_block(h, t_emb)

            # Apply attention if present
            if cross_attn is not None:
                # Project agent context to this decoder level
                agent_context_dec = F.interpolate(agent_context, size=h.shape[-2:], mode='bilinear', align_corners=False)
                agent_context_dec = self.decoder_context_projs[i](agent_context_dec)

                # ★ CRITICAL: Use checkpoint for attention to save memory!
                # Attention matrix can be 18GB+ for large spatial sizes
                if self.use_checkpoint and self.training:
                    h = checkpoint(cross_attn, h, agent_context_dec, use_reentrant=False)
                else:
                    h = cross_attn(h, agent_context_dec)

        # Output
        out = self.output_proj(h)
        return out


if __name__ == "__main__":
    # Test single-agent reconstruction
    model = HybridDiffusionUNet(
        in_channels=64,
        model_channels=64,
        out_channels=64,
        channel_mult=(1, 2, 4),
        latent_dim=8,
        time_emb_dim=256,
        num_decoder_attentions=1,  # 1 attention in decoder + 1 in bottleneck = 2 total
    )

    B = 2
    x_t = torch.randn(B, 64, 100, 352)
    t = torch.randint(0, 1000, (B,))

    # Single agent reconstruction (each agent reconstructs itself)
    agent_latents = [torch.randn(B, 8, 13, 44)]  # Only 1 agent
    transforms = [torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1)]  # Identity transform [B, 2, 3]

    with torch.no_grad():
        output = model(x_t, t, agent_latents, transforms)

    print(f"Input: {x_t.shape}")
    print(f"Output: {output.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Attention blocks: 1 (bottleneck) + {model.num_decoder_attentions} (decoder) = {1 + model.num_decoder_attentions}")
    print("✓ Single-agent reconstruction design verified")
