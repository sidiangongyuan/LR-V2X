"""
Modular Diffusion Architecture - Usage Examples

This demonstrates how to flexibly combine any fusion method with diffusion.

## How BEV Features are Extracted

The fusion models (HeterPyramidCollabMC, HeterModelBaselineMC, etc.) automatically
cache BEV features during their forward pass:

In fusion model forward():
    heter_feature_2d = torch.stack(heter_feature_2d_list)  # BEV features
    self._bev_features = heter_feature_2d  # Cached for diffusion

The FusionWithDiffusion wrapper then:
    1. Calls fusion_model.forward(data_dict)
    2. Extracts BEV features from fusion_model._bev_features
    3. Passes BEV features to diffusion module for training/reconstruction

This design ensures:
- Stage 1: Fusion model trains normally (BEV features cached but not used)
- Stage 2: Diffusion trains on frozen fusion's BEV features
- Stage 3: Both train together, diffusion learns to denoise fusion's features

Author: Claude
Date: 2025-01-26
"""

import torch
from opencood.models.heter_pyramid_collab_mc import HeterPyramidCollabMC
from opencood.models.heter_model_baseline_mc import MaxFusionMC, AttentionFusionMC  # Example
from opencood.models.fusion_with_diffusion_mc import FusionWithDiffusion


# ============================================================
# Example 1: Pyramid Fusion + Diffusion (Stage 3 Testing)
# ============================================================

def example1_pyramid_with_diffusion():
    """
    Load trained Pyramid (Stage 1) + trained Diffusion (Stage 2)
    → Test/Fine-tune in Stage 3
    """
    # Fusion config (from your pyramid yaml)
    fusion_config = {
        # ... your pyramid fusion config ...
        'num_class': 3,
        'fusion_backbone': {'in_channels': 64, ...},
        # ...
    }

    # Diffusion config
    diffusion_config = {
        'latent_dim': 32,
        'spatial_size': [100, 352],
        'num_timesteps': 1000,
        'beta_schedule': 'cosine',
        'use_mode': 'dit-b',
        'num_inference_steps': 20,
        'dit_config': {
            'patch_size': 4,
            'num_cross_attn_layers': 9
        }
    }

    # Create combined model
    model = FusionWithDiffusion(
        fusion_model_class=HeterPyramidCollabMC,
        fusion_args=fusion_config,
        diffusion_args=diffusion_config,
        use_diffusion=True
    )

    # Load checkpoints SEPARATELY
    model.load_fusion_checkpoint('stage1_pyramid_epoch30.pth')
    model.load_diffusion_checkpoint('stage2_diffusion_epoch25.pth')

    # Now ready for Stage 3!
    model.cuda()
    model.train()

    print("✅ Pyramid + Diffusion combined successfully!")


# ============================================================
# Example 2: Max Fusion + Same Diffusion (Reuse Diffusion!)
# ============================================================

def example2_max_with_diffusion():
    """
    Train Max Fusion (Stage 1) + Load Pyramid's Diffusion (Stage 2)
    → Test if diffusion generalizes to different fusion!
    """
    # Max fusion config
    max_config = {
        # ... your max fusion config ...
        'fusion_method': 'max',
        'num_class': 3,
        # ...
    }

    # Same diffusion config as before!
    diffusion_config = {
        'latent_dim': 32,
        'spatial_size': [100, 352],
        # ... same as Example 1 ...
    }

    # Create model with Max fusion
    model = FusionWithDiffusion(
        fusion_model_class=MaxFusionMC,  # Different fusion!
        fusion_args=max_config,
        diffusion_args=diffusion_config,
        use_diffusion=True
    )

    # Load DIFFERENT fusion but SAME diffusion
    model.load_fusion_checkpoint('stage1_max_epoch20.pth')
    model.load_diffusion_checkpoint('stage2_diffusion_epoch25.pth')  # Same as Example 1!

    model.cuda()
    model.eval()

    print("✅ Max Fusion + Reused Diffusion combined!")


# ============================================================
# Example 3: Stage-by-Stage Training
# ============================================================

def example3_stage_by_stage_training():
    """
    Train each stage separately with flexible checkpoint management.
    """
    # ========== Stage 1: Train Fusion Only ==========
    print("\n[Stage 1] Training Pyramid Fusion...")

    # Create model WITHOUT diffusion
    model_stage1 = FusionWithDiffusion(
        fusion_model_class=HeterPyramidCollabMC,
        fusion_args=fusion_config,
        diffusion_args=None,  # No diffusion!
        use_diffusion=False
    )

    # Train Stage 1
    optimizer = torch.optim.Adam(model_stage1.get_trainable_parameters('stage1'), lr=0.002)
    # ... training loop ...
    torch.save(model_stage1.fusion_model.state_dict(), 'stage1_pyramid.pth')


    # ========== Stage 2: Train Diffusion Only ==========
    print("\n[Stage 2] Training Diffusion...")

    # Create model WITH diffusion
    model_stage2 = FusionWithDiffusion(
        fusion_model_class=HeterPyramidCollabMC,
        fusion_args=fusion_config,
        diffusion_args=diffusion_config,
        use_diffusion=True
    )

    # Load Stage 1 fusion checkpoint
    model_stage2.load_fusion_checkpoint('stage1_pyramid.pth')

    # Freeze fusion parameters (critical!)
    model_stage2.freeze_fusion()

    # Get only diffusion parameters for optimizer
    diffusion_params = model_stage2.get_trainable_parameters('stage2')
    optimizer = torch.optim.Adam(diffusion_params, lr=0.0001)

    # Training loop
    for batch_data in train_loader:
        # Forward pass:
        # 1. Fusion model forward (frozen) produces BEV features
        # 2. BEV features are cached in fusion_model._bev_features
        # 3. Diffusion module reads _bev_features and computes reconstruction loss
        output_dict = model_stage2(batch_data)

        # Diffusion loss only
        diffusion_loss = output_dict['diffusion_loss']

        optimizer.zero_grad()
        diffusion_loss.backward()
        optimizer.step()

    # Save diffusion checkpoint
    torch.save(model_stage2.diffusion.state_dict(), 'stage2_diffusion.pth')


    # ========== Stage 3: Fine-tune All ==========
    print("\n[Stage 3] Fine-tuning Everything...")

    model_stage3 = FusionWithDiffusion(
        fusion_model_class=HeterPyramidCollabMC,
        fusion_args=fusion_config,
        diffusion_args=diffusion_config,
        use_diffusion=True
    )

    # Load both checkpoints
    model_stage3.load_fusion_checkpoint('stage1_pyramid.pth')
    model_stage3.load_diffusion_checkpoint('stage2_diffusion.pth')

    # Unfreeze everything
    all_params = model_stage3.get_trainable_parameters('stage3')
    optimizer = torch.optim.Adam(all_params, lr=0.0002)

    # ... training loop ...
    torch.save(model_stage3.state_dict(), 'stage3_final.pth')


# ============================================================
# Example 4: Mix and Match Different Combinations
# ============================================================

def example4_mix_and_match():
    """
    Test all combinations:
    - Pyramid + Diffusion
    - Max + Diffusion
    - Attention + Diffusion
    - Where2comm + Diffusion
    """
    fusion_methods = {
        'pyramid': HeterPyramidCollabMC,
        'max': MaxFusionMC,
        'attention': AttentionFusionMC,
        # Add more...
    }

    # Same diffusion for all!
    diffusion_checkpoint = 'stage2_diffusion_trained_on_pyramid.pth'

    results = {}

    for fusion_name, fusion_class in fusion_methods.items():
        print(f"\n{'='*60}")
        print(f"Testing: {fusion_name} + Diffusion")
        print('='*60)

        # Create model
        model = FusionWithDiffusion(
            fusion_model_class=fusion_class,
            fusion_args=configs[fusion_name],
            diffusion_args=diffusion_config,
            use_diffusion=True
        )

        # Load checkpoints
        model.load_fusion_checkpoint(f'stage1_{fusion_name}.pth')
        model.load_diffusion_checkpoint(diffusion_checkpoint)  # Same diffusion!

        # Evaluate
        model.cuda()
        model.eval()
        # ap = evaluate(model, val_loader)
        # results[fusion_name] = ap

        print(f"  ✅ {fusion_name} + Diffusion: AP@0.7 = ...")

    return results


# ============================================================
# Example 5: Checkpoint Organization
# ============================================================

"""
Recommended checkpoint directory structure:

checkpoints/
├── stage1_fusion/
│   ├── pyramid_epoch30.pth          # Pyramid fusion checkpoint
│   ├── max_epoch20.pth              # Max fusion checkpoint
│   ├── attention_epoch25.pth        # Attention fusion checkpoint
│   └── where2comm_epoch30.pth       # Where2comm fusion checkpoint
│
├── stage2_diffusion/
│   ├── diffusion_trained_on_pyramid.pth    # Diffusion trained with Pyramid
│   ├── diffusion_trained_on_max.pth        # Diffusion trained with Max
│   └── diffusion_universal.pth             # Diffusion trained on mixed data
│
└── stage3_combined/
    ├── pyramid+diffusion_epoch15.pth
    ├── max+diffusion_epoch15.pth
    └── ...

Loading example:
    # Any combination!
    model.load_fusion_checkpoint('checkpoints/stage1_fusion/max_epoch20.pth')
    model.load_diffusion_checkpoint('checkpoints/stage2_diffusion/diffusion_universal.pth')
"""


if __name__ == '__main__':
    # Run examples
    example1_pyramid_with_diffusion()
    example2_max_with_diffusion()
    # example3_stage_by_stage_training()
    # example4_mix_and_match()
