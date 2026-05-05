"""
DiT (Diffusion Transformer) for V2X Collaborative Perception

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
Official: https://github.com/facebookresearch/DiT

Adapted for V2X:
- Supports agent latents injection via cross-attention
- Handles BEV spatial features (not square images)
- Memory-efficient with configurable patch sizes
- AdaLN for time conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    """
    AdaLN modulation function.

    Args:
        x: Input tensor [B, N, C]
        shift: Shift parameter [B, C]
        scale: Scale parameter [B, C]

    Returns:
        Modulated tensor [B, N, C]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """
    Create 2D sinusoidal position embeddings.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid_size_h: Height in patches
        grid_size_w: Width in patches

    Returns:
        Position embeddings [H*W, embed_dim]
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # [2, H, W]

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])

    # Separate embeddings for H and W
    pos_embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, C/2]
    pos_embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, C/2]

    pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=1)  # [H*W, C]
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Create 1D sinusoidal embeddings.

    Args:
        embed_dim: Embedding dimension
        pos: Position grid [H, W]

    Returns:
        Embeddings [H*W, embed_dim]
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # [C/2]

    pos = pos.reshape(-1)  # [H*W]
    out = torch.einsum('m,d->md', pos, omega)  # [H*W, C/2]

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # [H*W, C]
    return emb


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding (supports non-square images).
    """
    def __init__(self, spatial_size=(100, 352), patch_size=4, in_chans=64, embed_dim=768):
        super().__init__()
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.grid_size = (spatial_size[0] // patch_size, spatial_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            patches: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: [B] timesteps
            dim: Embedding dimension

        Returns:
            [B, dim] embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t: [B] timesteps

        Returns:
            [B, hidden_size] embeddings
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AgentConditionEmbedder(nn.Module):
    """
    Embeds agent latents for conditioning.
    Processes agent latents to a single conditioning vector.
    """
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.latent_dim = latent_dim

        # Simple pooling + MLP to get conditioning vector
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, agent_latents):
        """
        Args:
            agent_latents: [B, latent_dim, h, w]

        Returns:
            condition: [B, hidden_size]
        """
        # Global average pooling
        B, C, H, W = agent_latents.shape
        pooled = agent_latents.mean(dim=[2, 3])  # [B, latent_dim]
        condition = self.mlp(pooled)  # [B, hidden_size]
        return condition


class TransformationEmbedder(nn.Module):
    """
    Embeds affine transformation matrix for spatial conditioning.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Affine matrix: [B, 2, 3] = 6 parameters
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_size // 2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size, bias=True),
        )

    def forward(self, affine_matrix):
        """
        Args:
            affine_matrix: [B, 2, 3] transformation matrix

        Returns:
            transform_emb: [B, hidden_size]
        """
        B = affine_matrix.size(0)
        flat = affine_matrix.view(B, -1)  # [B, 6]
        transform_emb = self.mlp(flat)  # [B, hidden_size]
        return transform_emb


class EgoBEVEncoder(nn.Module):
    """
    Encodes ego BEV features for conditioning.
    Uses patch embedding + pooling to get global context.
    """
    def __init__(self, in_channels, hidden_size, spatial_size=(100, 352), patch_size=4):
        super().__init__()
        # Patch embedding for ego BEV
        self.patch_embed = PatchEmbed(spatial_size, patch_size, in_channels, hidden_size)

        # Global pooling to get conditioning vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP to refine
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, ego_bev):
        """
        Args:
            ego_bev: [B, C, H, W] ego BEV features

        Returns:
            ego_cond: [B, hidden_size] global conditioning
            ego_patches: [B, N, hidden_size] patch-level features for cross-attention
        """
        # Patchify ego BEV
        ego_patches = self.patch_embed(ego_bev)  # [B, N, hidden_size]

        # Global pooling
        ego_pooled = self.pool(ego_patches.transpose(1, 2)).squeeze(-1)  # [B, hidden_size]

        # Refine
        ego_cond = self.mlp(ego_pooled)  # [B, hidden_size]

        return ego_cond, ego_patches


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Supports both self-attention and optional cross-attention with agent context.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_cross_attn=False,
                 latent_dim=32, use_checkpoint=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            # Project agent latent to hidden_size
            self.agent_proj = nn.Linear(latent_dim, hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # AdaLN modulation
        if use_cross_attn:
            # 9 parameters: shift/scale/gate for self-attn, cross-attn, mlp
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )
        else:
            # 6 parameters: shift/scale/gate for self-attn and mlp
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def _forward_impl(self, x, c, agent_context=None):
        """
        Implementation of forward pass.

        Args:
            x: [B, N, C] input patches
            c: [B, C] conditioning vector (time + agent global)
            agent_context: Optional [B, N_agent, C] for cross-attention

        Returns:
            [B, N, C] output patches
        """
        if self.use_cross_attn:
            # 9-way split
            shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(9, dim=1)
        else:
            # 6-way split
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        # Save attention weights if visualization is enabled
        need_weights = getattr(self, 'save_attn_weights', False)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=need_weights, average_attn_weights=False)
        if need_weights and attn_weights is not None:
            self.self_attn_weights = attn_weights.detach()
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Cross-attention (if agent context provided)
        if self.use_cross_attn and agent_context is not None:
            x_norm_ca = modulate(self.norm_cross(x), shift_ca, scale_ca)
            ca_out, ca_weights = self.cross_attn(x_norm_ca, agent_context, agent_context, need_weights=need_weights, average_attn_weights=False)
            if need_weights and ca_weights is not None:
                self.cross_attn_weights = ca_weights.detach()
            x = x + gate_ca.unsqueeze(1) * ca_out

        # MLP
        x_norm_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x

    def forward(self, x, c, agent_context=None):
        if self.use_checkpoint and self.training:
            # Use checkpoint to save memory
            return checkpoint(self._forward_impl, x, c, agent_context, use_reentrant=False)
        else:
            return self._forward_impl(x, c, agent_context)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: [B, N, C] patches
            c: [B, C] conditioning

        Returns:
            [B, N, patch_size^2 * out_channels]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_V2X(nn.Module):
    """
    Diffusion Transformer for V2X (DiT-V2X).

    Adapted from official DiT for V2X collaborative perception:
    - Handles non-square BEV features
    - Supports agent latents injection
    - Memory-efficient with checkpointing
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        spatial_size: tuple = (100, 352),
        patch_size: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_dim: int = 32,
        num_cross_attn_layers: int = 6,  # How many layers have cross-attention
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        # Patch embedding
        self.x_embedder = PatchEmbed(spatial_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches

        # Positional embedding (learnable or fixed)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Agent condition embedding (processes agent latent to conditioning vector)
        self.agent_embedder = AgentConditionEmbedder(latent_dim, hidden_size)

        # Ego BEV encoder (NEW: for ego condition)
        self.ego_embedder = EgoBEVEncoder(in_channels, hidden_size, spatial_size, patch_size)

        # Transformation embedder (NEW: for spatial relationship)
        self.transform_embedder = TransformationEmbedder(hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            # First num_cross_attn_layers have cross-attention
            use_cross = (i < num_cross_attn_layers)
            self.blocks.append(
                DiTBlock(hidden_size, num_heads, mlp_ratio, use_cross, latent_dim, use_checkpoint)
            )

        # For cross-attention: project agent latent patches
        self.agent_patch_proj = nn.Linear(latent_dim, hidden_size)

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        # Initialize
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.x_embedder.grid_size[0],
            self.x_embedder.grid_size[1]
        )
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize agent embedder
        nn.init.normal_(self.agent_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.agent_embedder.mlp[2].weight, std=0.02)

        # Initialize ego embedder (NEW)
        nn.init.normal_(self.ego_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.ego_embedder.mlp[2].weight, std=0.02)
        w = self.ego_embedder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.ego_embedder.patch_embed.proj.bias, 0)

        # Initialize transform embedder (NEW)
        nn.init.normal_(self.transform_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.transform_embedder.mlp[2].weight, std=0.02)

        # Zero-initialize adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-initialize final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Args:
            x: [B, N, patch_size^2 * C]

        Returns:
            imgs: [B, C, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h, w = self.x_embedder.grid_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        agent_latents: List[torch.Tensor],
        transforms: List[torch.Tensor],
        ego_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: Single-agent reconstruction from compressed latent.

        Args:
            x_t: [B, C, H, W] noisy input
            t: [B] timesteps
            agent_latents: List containing [B, latent_dim, h, w]
            transforms: List containing [B, 2, 3] affine matrix (agent->ego or ego->agent)
            ego_bev: Optional [B, C, H, W] ego BEV features for conditioning

        Returns:
            output: [B, C, H, W] denoised output
        """
        # Patchify
        x = self.x_embedder(x_t)  # [B, N, hidden_size]
        x = x + self.pos_embed  # Add positional embedding

        # Time embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]

        # Agent conditioning
        # Extract single agent latent (assume single agent for reconstruction)
        if agent_latents and len(agent_latents) > 0:
            z_agent = agent_latents[0]  # [B, latent_dim, h, w]

            # Global conditioning (pooled)
            agent_cond = self.agent_embedder(z_agent)  # [B, hidden_size]

            # Cross-attention context (patchified)
            z_patches = z_agent.flatten(2).transpose(1, 2)  # [B, h*w, latent_dim]
            agent_context = self.agent_patch_proj(z_patches)  # [B, h*w, hidden_size]
        else:
            agent_cond = torch.zeros_like(t_emb)
            agent_context = None

        # Ego BEV conditioning (NEW)
        if ego_bev is not None:
            ego_cond, ego_patches = self.ego_embedder(ego_bev)  # [B, hidden_size], [B, N, hidden_size]
            # Combine ego patches with agent context for cross-attention
            if agent_context is not None:
                # Concatenate along sequence dimension
                agent_context = torch.cat([agent_context, ego_patches], dim=1)  # [B, h*w+N, hidden_size]
            else:
                agent_context = ego_patches
        else:
            ego_cond = torch.zeros_like(t_emb)

        # Transformation conditioning (NEW)
        if transforms and len(transforms) > 0:
            T = transforms[0]  # [B, 2, 3]
            transform_cond = self.transform_embedder(T)  # [B, hidden_size]
        else:
            transform_cond = torch.zeros_like(t_emb)

        # Combined conditioning
        c = t_emb + agent_cond + ego_cond + transform_cond  # [B, hidden_size]

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, agent_context)

        # Final layer
        x = self.final_layer(x, c)  # [B, N, patch_size^2 * out_channels]

        # Unpatchify
        x = self.unpatchify(x)  # [B, out_channels, H, W]

        return x


def DiT_V2X_S(**kwargs):
    """DiT-V2X Small: depth=12, hidden_size=384, num_heads=6"""
    return DiT_V2X(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_V2X_B(**kwargs):
    """DiT-V2X Base: depth=12, hidden_size=768, num_heads=12"""
    return DiT_V2X(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_V2X_L(**kwargs):
    """DiT-V2X Large: depth=24, hidden_size=1024, num_heads=16"""
    return DiT_V2X(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_V2X_XL(**kwargs):
    """DiT-V2X XLarge: depth=28, hidden_size=1152, num_heads=16"""
    return DiT_V2X(depth=28, hidden_size=1152, num_heads=16, **kwargs)


if __name__ == "__main__":
    # Test DiT-V2X
    print("Testing DiT-V2X...")

    model = DiT_V2X_B(
        in_channels=64,
        out_channels=64,
        spatial_size=(100, 352),
        patch_size=4,
        latent_dim=32,
        num_cross_attn_layers=6,
        use_checkpoint=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward
    B = 2
    x_t = torch.randn(B, 64, 100, 352)
    t = torch.randint(0, 1000, (B,))
    agent_latents = [torch.randn(B, 32, 13, 44)]
    transforms = [torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1)]
    ego_bev = torch.randn(B, 64, 100, 352)  # NEW: ego BEV condition

    with torch.no_grad():
        # Test with ego condition
        output = model(x_t, t, agent_latents, transforms, ego_bev=ego_bev)
        print(f"Input shape: {x_t.shape}")
        print(f"Output shape: {output.shape}")

        # Test without ego condition (backward compatibility)
        output_no_ego = model(x_t, t, agent_latents, transforms, ego_bev=None)
        print(f"Output shape (no ego): {output_no_ego.shape}")

    print("DiT-V2X test passed!")
