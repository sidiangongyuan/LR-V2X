"""
Compute BEV Statistical Prior from Training Set

This script extracts BEV features from all training samples and computes
the mean and standard deviation to use as prior for diffusion.

Usage:
    python opencood/tools/compute_bev_prior.py \
        --hypes_yaml opencood/hypes_yaml/v2x_real/DiffV2X_Stages/diffv2x_stage1.yaml \
        --model_dir /path/to/stage1/checkpoint \
        --output_path bev_prior_statistics.pth
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def test_parser():
    parser = argparse.ArgumentParser(description="Compute BEV prior from training set")
    parser.add_argument('--hypes_yaml', type=str, required=True,
                        help='Config file for stage 1 model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to stage 1 checkpoint directory')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for prior statistics (default: save to model_dir/bev_prior_statistics.pth)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (default: use all)')

    opt = parser.parse_args()

    # Auto-set output path if not specified
    if opt.output_path is None:
        opt.output_path = os.path.join(opt.model_dir, 'bev_prior_statistics.pth')
        print(f"[Auto] Output path set to: {opt.output_path}\n")

    return opt


def extract_bev_features(batch_data, model):
    """
    Extract BEV features before fusion.

    Args:
        batch_data: Batch from dataloader
        model: Stage 1 model

    Returns:
        bev_features: [N_agents, C, H, W] BEV features
    """
    with torch.no_grad():
        # IMPORTANT: Keep use_diffusion=True if model has diffusion
        # because _original_bev_features is only set when use_diffusion=True
        # We just won't compute diffusion loss during BEV prior extraction

        try:
            # Forward pass - will extract BEV features internally
            output = model(batch_data['ego'])

            # Extract the BEV features before fusion
            # Different models may store this differently:
            # - diffv2x_pyramid_mc: _original_bev_features (when use_diffusion=True)
            # - heter_pyramid_collab_mc: _bev_features
            if hasattr(model, '_original_bev_features') and model._original_bev_features is not None:
                bev_features = model._original_bev_features
            elif hasattr(model, '_bev_features') and model._bev_features is not None:
                bev_features = model._bev_features
            elif hasattr(model, 'heter_feature_2d') and model.heter_feature_2d is not None:
                bev_features = model.heter_feature_2d
            else:
                # Fallback: try to extract from modality-specific encoders
                modality_name = 'm1'
                if hasattr(model, f'encoder_{modality_name}'):
                    encoder = getattr(model, f'encoder_{modality_name}')
                    backbone = getattr(model, f'backbone_{modality_name}')

                    # Get input data - check multiple possible locations
                    ego_data = batch_data['ego']

                    # Check for voxel data in multiple possible locations
                    # Try 1: processed_lidar dict
                    if 'processed_lidar' in ego_data and isinstance(ego_data['processed_lidar'], dict):
                        voxel_features = ego_data['processed_lidar']['voxel_features']
                        voxel_coords = ego_data['processed_lidar']['voxel_coords']
                        voxel_num_points = ego_data['processed_lidar']['voxel_num_points']
                    # Try 2: top level
                    elif 'voxel_features' in ego_data:
                        voxel_features = ego_data['voxel_features']
                        voxel_coords = ego_data['voxel_coords']
                        voxel_num_points = ego_data['voxel_num_points']
                    # Try 3: Check for modality-specific keys (m1, m2, etc.)
                    elif 'm1' in ego_data and isinstance(ego_data['m1'], dict):
                        if 'processed_lidar' in ego_data['m1']:
                            voxel_features = ego_data['m1']['processed_lidar']['voxel_features']
                            voxel_coords = ego_data['m1']['processed_lidar']['voxel_coords']
                            voxel_num_points = ego_data['m1']['processed_lidar']['voxel_num_points']
                        elif 'voxel_features' in ego_data['m1']:
                            voxel_features = ego_data['m1']['voxel_features']
                            voxel_coords = ego_data['m1']['voxel_coords']
                            voxel_num_points = ego_data['m1']['voxel_num_points']
                        else:
                            raise KeyError(f"Cannot find voxel data in batch. Available keys: {list(ego_data.keys())}")
                    else:
                        raise KeyError(f"Cannot find voxel data in batch. Available keys: {list(ego_data.keys())}")

                    record_len = ego_data['record_len']

                    # Encode
                    batch_dict = {
                        'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len
                    }
                    batch_dict = encoder(batch_dict)

                    # Backbone
                    bev_features = backbone(batch_dict)['spatial_features_2d']
                else:
                    raise ValueError(f"Cannot extract BEV features from model")

            return bev_features

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to extract BEV features: {e}") from e


def main():
    opt = test_parser()

    print(f"{'='*80}")
    print(f"Computing BEV Statistical Prior")
    print(f"{'='*80}")
    print(f"Config: {opt.hypes_yaml}")
    print(f"Model: {opt.model_dir}")
    print(f"Output: {opt.output_path}")
    print(f"{'='*80}\n")

    # Load config
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # Build dataset (training set)
    print("Building training dataset...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"Training set size: {len(train_dataset)}")

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one batch at a time
        num_workers=4,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    # Build model
    print("\nBuilding model...")
    model = train_utils.create_model(hypes)

    # Load checkpoint
    checkpoint_path = os.path.join(opt.model_dir, 'net_epoch_bestval_at30.pth')
    if not os.path.exists(checkpoint_path):
        # Try to find any checkpoint
        checkpoints = [f for f in os.listdir(opt.model_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoint_path = os.path.join(opt.model_dir, checkpoints[0])
        else:
            raise FileNotFoundError(f"No checkpoint found in {opt.model_dir}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        # Format: {'net': state_dict, 'epoch': ..., ...}
        state_dict = checkpoint['net']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Format: {'model_state_dict': state_dict, ...}
        state_dict = checkpoint['model_state_dict']
    else:
        # Direct state_dict
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}\n")

    # Collect BEV features
    print("Extracting BEV features from training set...")
    all_bev_features = []

    max_samples = opt.max_samples if opt.max_samples else len(train_loader)

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(train_loader, total=min(max_samples, len(train_loader)))):
            if i >= max_samples:
                break

            # Move data to device
            batch_data = train_utils.to_device(batch_data, device)

            try:
                # Extract BEV features
                bev_features = extract_bev_features(batch_data, model)
                # bev_features shape: [N_agents, C, H, W]

                # Move to CPU and store
                all_bev_features.append(bev_features.cpu())

            except Exception as e:
                print(f"\nWarning: Failed to process batch {i}: {e}")
                continue

    # Concatenate all features
    print("\nComputing statistics...")
    all_bev_features = torch.cat(all_bev_features, dim=0)  # [Total_agents, C, H, W]
    print(f"Total BEV features collected: {all_bev_features.shape[0]}")
    print(f"BEV feature shape: {all_bev_features.shape}")

    # Compute statistics
    bev_mean = all_bev_features.mean(dim=0)  # [C, H, W]
    bev_std = all_bev_features.std(dim=0)    # [C, H, W]

    # Statistics
    print(f"\nStatistics:")
    print(f"  BEV mean range: [{bev_mean.min():.4f}, {bev_mean.max():.4f}]")
    print(f"  BEV std range:  [{bev_std.min():.4f}, {bev_std.max():.4f}]")
    print(f"  BEV mean (avg): {bev_mean.mean():.4f}")
    print(f"  BEV std (avg):  {bev_std.mean():.4f}")

    # Save
    output_data = {
        'bev_mean': bev_mean,
        'bev_std': bev_std,
        'num_samples': len(all_bev_features),
        'shape': list(bev_mean.shape),
        'config_path': opt.hypes_yaml,
        'model_path': opt.model_dir,
    }

    torch.save(output_data, opt.output_path)
    print(f"\n✓ Prior statistics saved to: {opt.output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
