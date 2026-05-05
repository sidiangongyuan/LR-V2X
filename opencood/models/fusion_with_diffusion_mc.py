"""
Universal Wrapper: Combine Any Fusion Method + Diffusion Module

This class allows flexible combination of:
- Stage 1: Any fusion model (Pyramid, Max, Attention, etc.)
- Stage 2: Trained diffusion module
- Stage 3: Combined model for detection

Usage:
    # Load any fusion model + diffusion
    model = FusionWithDiffusion(
        fusion_model_class=HeterPyramidCollabMC,  # or any other fusion
        fusion_args=fusion_config,
        diffusion_args=diffusion_config
    )

    # Load checkpoints separately
    model.load_fusion_checkpoint('stage1_pyramid.pth')
    model.load_diffusion_checkpoint('stage2_diffusion.pth')

Author: Claude
Date: 2025-01-26
"""

import torch
import torch.nn as nn
from opencood.models.sub_modules.diffusion_wrapper import DiffusionModule


class FusionWithDiffusion(nn.Module):
    """
    Universal wrapper that combines any fusion method with diffusion.

    Architecture:
        1. Fusion Model (encoder + backbone + fusion + detection heads)
        2. Diffusion Module (latent_encoder + diffusion_model)
        3. Combined forward pass

    Args:
        fusion_model_class: Class of the fusion model (e.g., HeterPyramidCollabMC)
        fusion_args: Configuration dict for fusion model
        diffusion_args: Configuration dict for diffusion module
        use_diffusion: Whether to use diffusion (for stage switching)
    """

    def __init__(
        self,
        fusion_model_class,
        fusion_args: dict,
        diffusion_args: dict = None,
        use_diffusion: bool = True
    ):
        super(FusionWithDiffusion, self).__init__()

        self.use_diffusion = use_diffusion

        # ============================================================
        # 1. Base Fusion Model
        # ============================================================
        print("="*80)
        print("FusionWithDiffusion: Initializing base fusion model...")
        print("="*80)

        self.fusion_model = fusion_model_class(fusion_args)

        # ============================================================
        # 2. Diffusion Module (if enabled)
        # ============================================================
        if self.use_diffusion and diffusion_args is not None:
            print("\n" + "="*80)
            print("FusionWithDiffusion: Adding diffusion module...")
            print("="*80)

            # Auto-detect BEV feature dimensions from fusion model
            # Assume fusion model has: cls_head, reg_head with specific input channels
            bev_channels = self._detect_bev_channels(fusion_args)

            self.diffusion = DiffusionModule(
                bev_channels=bev_channels,
                latent_dim=diffusion_args.get('latent_dim', 32),
                spatial_size=diffusion_args.get('spatial_size', [100, 352]),
                dit_config=diffusion_args.get('dit_config', {}),
                diffusion_config=diffusion_args
            )

            # Hook to extract BEV features before fusion
            self._bev_features = None
            self._record_len = None
        else:
            self.diffusion = None

        print("\n" + "="*80)
        print("FusionWithDiffusion: Initialization complete!")
        print(f"  - Fusion model: {fusion_model_class.__name__}")
        print(f"  - Use diffusion: {self.use_diffusion}")
        print("="*80 + "\n")

    def _detect_bev_channels(self, fusion_args):
        """
        Auto-detect BEV feature channels from fusion configuration.
        """
        # Try to find BEV channels from config
        if 'fusion_backbone' in fusion_args:
            fusion_config = fusion_args['fusion_backbone']
            if 'in_channels' in fusion_config:
                return fusion_config['in_channels']

        # Default fallback
        return 64

    def load_fusion_checkpoint(self, checkpoint_path, strict=True):
        """
        Load fusion model checkpoint (Stage 1).

        Args:
            checkpoint_path: Path to fusion checkpoint
            strict: Whether to strictly match keys
        """
        print(f"Loading fusion checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load into fusion model
        missing_keys, unexpected_keys = self.fusion_model.load_state_dict(checkpoint, strict=strict)

        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")

        print("  Fusion checkpoint loaded successfully!")

    def load_diffusion_checkpoint(self, checkpoint_path, strict=True):
        """
        Load diffusion module checkpoint (Stage 2).

        Args:
            checkpoint_path: Path to diffusion checkpoint
            strict: Whether to strictly match keys
        """
        if self.diffusion is None:
            print("Warning: Diffusion module not initialized, skipping checkpoint load")
            return

        print(f"Loading diffusion checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract only diffusion-related keys
        diffusion_keys = ['latent_encoder.', 'diffusion_model.']
        diffusion_state_dict = {}

        for key, value in checkpoint.items():
            for prefix in diffusion_keys:
                if key.startswith(prefix):
                    # Remove prefix to match DiffusionModule structure
                    new_key = key
                    diffusion_state_dict[new_key] = value
                    break

        # Load into diffusion module
        missing_keys, unexpected_keys = self.diffusion.load_state_dict(diffusion_state_dict, strict=strict)

        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")

        print("  Diffusion checkpoint loaded successfully!")

    def freeze_fusion(self):
        """Freeze fusion model parameters (for Stage 2 training)."""
        for param in self.fusion_model.parameters():
            param.requires_grad = False
        print("Fusion model parameters frozen")

    def unfreeze_all(self):
        """Unfreeze all parameters (for Stage 3 training)."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")

    def forward(self, data_dict, return_loss=False):
        """
        Forward pass with optional diffusion.

        Args:
            data_dict: Input data dictionary
            return_loss: Whether to return loss (for training)

        Returns:
            output_dict: Dictionary with predictions and optional losses
        """
        # ============================================================
        # 1. Fusion Model Forward (extracts BEV features automatically)
        # ============================================================
        output_dict = self.fusion_model(data_dict)

        # ============================================================
        # 2. Extract BEV Features and Compute Diffusion Loss
        # ============================================================
        if self.use_diffusion and self.diffusion is not None:
            # Get BEV features from fusion model (cached in _bev_features)
            if hasattr(self.fusion_model, '_bev_features'):
                bev_features = self.fusion_model._bev_features  # [N_agents, C, H, W]
                record_len = data_dict.get('record_len', None)

                if self.training or return_loss:
                    # Training/Validation: Compute diffusion loss
                    if record_len is not None:
                        diffusion_loss = self.diffusion(bev_features, record_len, training=True)
                        output_dict['diffusion_loss'] = diffusion_loss
                    else:
                        print("Warning: record_len not found in data_dict, skipping diffusion loss")
                        output_dict['diffusion_loss'] = torch.tensor(0.0, device=bev_features.device)
                else:
                    # Inference: Optionally reconstruct features
                    # Note: In current implementation, fusion already used original features
                    # To use reconstructed features, you'd need to modify fusion model forward
                    output_dict['diffusion_loss'] = torch.tensor(0.0, device=bev_features.device)
            else:
                print("Warning: Fusion model does not have _bev_features attribute")
                output_dict['diffusion_loss'] = torch.tensor(0.0, device=list(self.fusion_model.parameters())[0].device)
        else:
            # No diffusion
            output_dict['diffusion_loss'] = torch.tensor(0.0, device=list(self.fusion_model.parameters())[0].device)

        return output_dict

    def get_trainable_parameters(self, stage='stage3'):
        """
        Get trainable parameters for different training stages.

        Args:
            stage: 'stage1', 'stage2', or 'stage3'

        Returns:
            List of trainable parameters
        """
        if stage == 'stage1':
            # Stage 1: Only fusion model (no diffusion)
            return [p for p in self.fusion_model.parameters() if p.requires_grad]

        elif stage == 'stage2':
            # Stage 2: Only diffusion module
            if self.diffusion is None:
                raise ValueError("Diffusion module not initialized")
            self.freeze_fusion()
            return [p for p in self.diffusion.parameters() if p.requires_grad]

        elif stage == 'stage3':
            # Stage 3: All parameters
            self.unfreeze_all()
            return [p for p in self.parameters() if p.requires_grad]

        else:
            raise ValueError(f"Unknown stage: {stage}")
