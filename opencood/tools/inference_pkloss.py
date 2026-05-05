# -*- coding: utf-8 -*-
# Author: Based on inference.py
# Purpose: Test communication packet loss robustness
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Inference script to test model robustness under communication packet loss.
Tests multiple compression ratios: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
Saves results to CSV table for easy comparison.
"""

import sys
import argparse
import copy
import os
from typing import Any, Dict, List

# Allow running this file directly without installing the package (e.g., without `pip install -e .`).
# Keep consistent with `opencood/tools/diffv2x_stages/*.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import random
import time
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import simple_vis


torch.multiprocessing.set_sharing_strategy('file_system')


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def seed_all(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_parser():
    parser = argparse.ArgumentParser(description="Packet Loss Robustness Testing")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional directory to save evaluation outputs. Defaults to model_dir.')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='Fusion method: no, late, early, or intermediate')
    parser.add_argument('--save_vis', action='store_true',
                        help='Whether to save visualization results')
    parser.add_argument('--save_vis_interval', type=int, default=50,
                        help='Interval of saving visualization (if enabled)')
    parser.add_argument('--sample_indices', nargs='+', type=int, default=None,
                        help='Optional dataset indices for qualitative visualization. '
                             'When set, inference runs only on these samples and saved '
                             'filenames keep the original dataset indices.')
    parser.add_argument('--vis_tag', type=str, default='',
                        help='Optional visualization subdirectory name. Overrides the default vis directory when set.')
    parser.add_argument('--inference_timestep', type=int, default=None,
                        help='Optional override for diffusion inference timestep.')
    parser.add_argument('--note', default="", type=str, help='Any note')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Packet loss ratios to test
    parser.add_argument('--test_ratios', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
                        help='Compression ratios to test (0.1 = 90%% packet loss)')
    parser.add_argument('--packet_loss_mode', type=str, default='bernoulli',
                        choices=['bernoulli', 'burst'],
                        help='Packet loss pattern: independent Bernoulli or spatially-correlated burst loss.')
    parser.add_argument('--burst_coarse_h', type=int, default=8,
                        help='Coarse burst grid height before nearest-neighbor upsampling.')
    parser.add_argument('--burst_coarse_w', type=int, default=16,
                        help='Coarse burst grid width before nearest-neighbor upsampling.')
    parser.add_argument('--packet_loss_seed_base', type=int, default=None,
                        help='Optional base seed for deterministic per-sample/per-agent packet loss masks.')

    opt = parser.parse_args()
    return opt


def _build_packet_loss_suffix(opt: argparse.Namespace) -> str:
    if opt.packet_loss_mode == 'bernoulli' and opt.packet_loss_seed_base is None:
        return ''

    suffix_parts = [f"_{opt.packet_loss_mode}"]
    if opt.packet_loss_mode == 'burst':
        suffix_parts.append(f"_c{int(opt.burst_coarse_h)}x{int(opt.burst_coarse_w)}")
    if opt.packet_loss_seed_base is not None:
        suffix_parts.append(f"_seed{int(opt.packet_loss_seed_base)}")
    return ''.join(suffix_parts)


def _get_output_dir(opt: argparse.Namespace) -> str:
    output_dir = opt.save_dir if opt.save_dir else opt.model_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _normalize_sample_indices(sample_indices, dataset_len: int) -> List[int]:
    if not sample_indices:
        return []

    normalized = []
    seen = set()
    for idx in sample_indices:
        idx = int(idx)
        if idx < 0 or idx >= dataset_len:
            raise ValueError(f'sample index {idx} out of range [0, {dataset_len - 1}]')
        if idx in seen:
            continue
        normalized.append(idx)
        seen.add(idx)
    return normalized


def _packet_loss_overrides(
    compression_ratio: float,
    opt: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        'compression_ratio': float(compression_ratio),
        'packet_loss_mode': opt.packet_loss_mode,
        'burst_coarse_h': int(opt.burst_coarse_h),
        'burst_coarse_w': int(opt.burst_coarse_w),
        'packet_loss_seed_base': opt.packet_loss_seed_base,
    }


def _apply_model_arg_overrides(model_args: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    touched_paths: List[str] = []

    for key, value in overrides.items():
        model_args[key] = value
    touched_paths.append('model.args')

    for sub_key, sub_value in model_args.items():
        if not isinstance(sub_value, dict):
            continue
        for key, value in overrides.items():
            sub_value[key] = value
        touched_paths.append(f'model.args.{sub_key}')

    return touched_paths


def _override_inference_timestep_if_present(model_args: Dict[str, Any], inference_timestep: int) -> List[str]:
    touched_paths: List[str] = []
    diffusion_args = model_args.get('diffusion', None)
    if isinstance(diffusion_args, dict):
        diffusion_args['inference_timestep'] = int(inference_timestep)
        touched_paths.append('model.args.diffusion.inference_timestep')
    return touched_paths


def load_model_with_packet_loss_config(
    hypes_base: Dict[str, Any],
    compression_ratio: float,
    saved_path: str,
    device: torch.device,
    opt: argparse.Namespace,
):
    """
    Load model with specified compression_ratio.

    This function automatically detects where to set compression_ratio
    by traversing the model configuration.

    Args:
        hypes: Configuration dictionary
        compression_ratio: Float in [0, 1], ratio of packets to keep
        saved_path: Path to saved model checkpoint
        device: torch device

    Returns:
        model: Loaded model with compression_ratio set
    """
    hypes = copy.deepcopy(hypes_base)
    model_args = hypes.get('model', {}).get('args', None)
    if not isinstance(model_args, dict):
        raise KeyError('Could not find model.args in config; packet loss overrides cannot be applied.')

    overrides = _packet_loss_overrides(compression_ratio, opt)
    touched_paths = _apply_model_arg_overrides(model_args, overrides)
    if opt.inference_timestep is not None:
        touched_paths.extend(
            _override_inference_timestep_if_present(model_args, opt.inference_timestep)
        )

    print(
        "[Config] Packet loss overrides: "
        f"ratio={compression_ratio:.2f}, mode={opt.packet_loss_mode}, "
        f"burst=({int(opt.burst_coarse_h)}, {int(opt.burst_coarse_w)}), "
        f"seed_base={opt.packet_loss_seed_base}"
    )
    if opt.inference_timestep is not None:
        if any(path.endswith('inference_timestep') for path in touched_paths):
            print(f"[Config] inference_timestep override: {opt.inference_timestep}")
        else:
            print("[Warning] --inference_timestep is set but no diffusion config was found.")
    print(f"[Config] Updated {len(touched_paths)} location(s): {', '.join(touched_paths)}")

    # Create model
    model = train_utils.create_model(hypes)
    device = torch.device(device)

    # Load checkpoint
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f'Loading Model from checkpoint at epoch {resume_epoch}')

    model = model.to(device)
    model.eval()

    return model


def inference_single_ratio(
    opt,
    hypes,
    compression_ratio,
    device,
    opencood_dataset,
    eval_dataset=None,
    dataset_index_lookup=None,
):
    """
    Run inference for a single compression ratio.

    Returns:
        results_dict: Dictionary with AP@0.3, AP@0.5, AP@0.7 for all ranges
    """
    print("\n" + "="*80)
    print(f"Testing Compression Ratio: {compression_ratio:.2f} ({(1-compression_ratio)*100:.1f}% packet loss)")
    print("="*80)

    # Load model with this compression ratio
    model = load_model_with_packet_loss_config(
        hypes,
        compression_ratio,
        opt.model_dir,
        device,
        opt,
    )

    # Create data loader
    if eval_dataset is None:
        eval_dataset = opencood_dataset

    data_loader = DataLoader(eval_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

    # Create result statistics for different ranges
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_short = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                         0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                         0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_middle = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                          0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                          0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_long = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    total_time = 0.0
    num_batches = 0

    # Inference loop
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader, desc=f"Ratio={compression_ratio:.1f}")):
            if batch_data is None:
                continue
            dataset_index = dataset_index_lookup[i] if dataset_index_lookup is not None else i

            batch_data = train_utils.to_device(batch_data, device)

            # Run inference based on fusion method
            _synchronize_if_needed(device)
            start_time = time.time()
            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            else:
                raise NotImplementedError(f'Fusion method {opt.fusion_method} not supported')
            _synchronize_if_needed(device)
            batch_time = time.time() - start_time
            total_time += batch_time
            num_batches += 1

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            # Calculate TP/FP for different IoU thresholds and distance ranges
            for iou_threshold in [0.3, 0.5, 0.7]:
                # Overall
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                          pred_score,
                                          gt_box_tensor,
                                          result_stat,
                                          iou_threshold)
                # Short range: 0-30m
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                          pred_score,
                                          gt_box_tensor,
                                          result_stat_short,
                                          iou_threshold,
                                          left_range=0,
                                          right_range=30)
                # Middle range: 30-50m
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                          pred_score,
                                          gt_box_tensor,
                                          result_stat_middle,
                                          iou_threshold,
                                          left_range=30,
                                          right_range=50)
                # Long range: 50-100m
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                          pred_score,
                                          gt_box_tensor,
                                          result_stat_long,
                                          iou_threshold,
                                          left_range=50,
                                          right_range=100)

            # Optional visualization
            should_save_vis = False
            if opt.save_vis:
                if dataset_index_lookup is not None:
                    should_save_vis = True
                else:
                    should_save_vis = (i % opt.save_vis_interval == 0)

            if should_save_vis:
                output_dir = _get_output_dir(opt)
                vis_suffix = _build_packet_loss_suffix(opt)
                vis_dir_name = opt.vis_tag if opt.vis_tag else f'vis_pkloss_{compression_ratio:.1f}{vis_suffix}'
                vis_save_path = os.path.join(output_dir, vis_dir_name)
                os.makedirs(vis_save_path, exist_ok=True)
                vis_file = os.path.join(vis_save_path, f'{dataset_index:05d}.png')
                simple_vis.visualize(
                    infer_result,
                    batch_data['ego']['origin_lidar'][0],
                    hypes['postprocess']['gt_range'],
                    vis_file,
                    method='bev',
                    left_hand=False,
                )

            torch.cuda.empty_cache()

    # Calculate final AP for all ranges
    ap_30, _, _ = eval_utils.calculate_ap(result_stat, 0.30)
    ap_50, _, _ = eval_utils.calculate_ap(result_stat, 0.50)
    ap_70, _, _ = eval_utils.calculate_ap(result_stat, 0.70)

    ap_30_short, _, _ = eval_utils.calculate_ap(result_stat_short, 0.30)
    ap_50_short, _, _ = eval_utils.calculate_ap(result_stat_short, 0.50)
    ap_70_short, _, _ = eval_utils.calculate_ap(result_stat_short, 0.70)

    ap_30_middle, _, _ = eval_utils.calculate_ap(result_stat_middle, 0.30)
    ap_50_middle, _, _ = eval_utils.calculate_ap(result_stat_middle, 0.50)
    ap_70_middle, _, _ = eval_utils.calculate_ap(result_stat_middle, 0.70)

    ap_30_long, _, _ = eval_utils.calculate_ap(result_stat_long, 0.30)
    ap_50_long, _, _ = eval_utils.calculate_ap(result_stat_long, 0.50)
    ap_70_long, _, _ = eval_utils.calculate_ap(result_stat_long, 0.70)

    print(f"\nResults for compression_ratio={compression_ratio:.2f}:")
    print(f"  Overall - AP@0.3: {ap_30:.4f}, AP@0.5: {ap_50:.4f}, AP@0.7: {ap_70:.4f}")
    print(f"  Short   - AP@0.3: {ap_30_short:.4f}, AP@0.5: {ap_50_short:.4f}, AP@0.7: {ap_70_short:.4f}")
    print(f"  Middle  - AP@0.3: {ap_30_middle:.4f}, AP@0.5: {ap_50_middle:.4f}, AP@0.7: {ap_70_middle:.4f}")
    print(f"  Long    - AP@0.3: {ap_30_long:.4f}, AP@0.5: {ap_50_long:.4f}, AP@0.7: {ap_70_long:.4f}")

    if num_batches > 0 and total_time > 0:
        inference_only_fps = num_batches / total_time
        avg_inference_time_ms = total_time / num_batches * 1000.0
        print(f"  FPS     - Inference-only: {inference_only_fps:.2f} frames/s")
        print(f"  Time    - Avg per frame: {avg_inference_time_ms:.2f} ms")
    else:
        inference_only_fps = float('nan')
        avg_inference_time_ms = float('nan')
        print("  FPS     - Inference-only: N/A")
        print("  Time    - Avg per frame: N/A")

    # Clean up model
    del model
    torch.cuda.empty_cache()

    return {
        'compression_ratio': compression_ratio,
        'packet_loss_pct': (1 - compression_ratio) * 100,
        'packet_loss_mode': opt.packet_loss_mode,
        'burst_coarse_h': int(opt.burst_coarse_h),
        'burst_coarse_w': int(opt.burst_coarse_w),
        'packet_loss_seed_base': opt.packet_loss_seed_base,
        'FPS (Inference)': inference_only_fps,
        'Avg Time (ms)': avg_inference_time_ms,
        # Overall
        'AP@0.3': ap_30,
        'AP@0.5': ap_50,
        'AP@0.7': ap_70,
        # Short range
        'AP@0.3_short': ap_30_short,
        'AP@0.5_short': ap_50_short,
        'AP@0.7_short': ap_70_short,
        # Middle range
        'AP@0.3_middle': ap_30_middle,
        'AP@0.5_middle': ap_50_middle,
        'AP@0.7_middle': ap_70_middle,
        # Long range
        'AP@0.3_long': ap_30_long,
        'AP@0.5_long': ap_50_long,
        'AP@0.7_long': ap_70_long,
    }


def main():
    opt = test_parser()

    # Set random seed
    seed_all(opt.seed)
    print(f"Random seed set to: {opt.seed}")

    # Load configuration
    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=bool(opt.save_vis), train=False)
    eval_dataset = opencood_dataset
    dataset_index_lookup = None

    selected_indices = _normalize_sample_indices(opt.sample_indices, len(opencood_dataset))
    if selected_indices:
        print(f'[Qualitative] Restricting inference to sample indices: {selected_indices}')
        print('[Qualitative] Reported metrics will only reflect this subset.')
        eval_dataset = Subset(opencood_dataset, selected_indices)
        dataset_index_lookup = selected_indices

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test all compression ratios
    all_results = []

    for compression_ratio in opt.test_ratios:
        result = inference_single_ratio(
            opt,
            hypes,
            compression_ratio,
            device,
            opencood_dataset,
            eval_dataset=eval_dataset,
            dataset_index_lookup=dataset_index_lookup,
        )
        all_results.append(result)

    # ============================================================
    # Save results to CSV
    # ============================================================
    df_results = pd.DataFrame(all_results)

    # Reorder columns for better readability
    cols_order = ['compression_ratio', 'packet_loss_pct',
                  'packet_loss_mode', 'burst_coarse_h', 'burst_coarse_w', 'packet_loss_seed_base',
                  'FPS (Inference)', 'Avg Time (ms)',
                  'AP@0.3', 'AP@0.5', 'AP@0.7',
                  'AP@0.3_short', 'AP@0.5_short', 'AP@0.7_short',
                  'AP@0.3_middle', 'AP@0.5_middle', 'AP@0.7_middle',
                  'AP@0.3_long', 'AP@0.5_long', 'AP@0.7_long']
    df_results = df_results[cols_order]

    # Format for better readability
    df_results['compression_ratio'] = df_results['compression_ratio'].apply(lambda x: f'{x:.2f}')
    df_results['packet_loss_pct'] = df_results['packet_loss_pct'].apply(lambda x: f'{x:.1f}%')
    df_results['packet_loss_seed_base'] = df_results['packet_loss_seed_base'].apply(
        lambda x: '' if pd.isna(x) else str(int(x))
    )
    for col in ['FPS (Inference)', 'Avg Time (ms)']:
        df_results[col] = df_results[col].apply(
            lambda x: '' if pd.isna(x) else f'{x:.2f}'
        )

    # Format all AP columns
    ap_cols = [col for col in df_results.columns if col.startswith('AP@')]
    for col in ap_cols:
        df_results[col] = df_results[col].apply(lambda x: f'{x:.4f}')

    output_dir = _get_output_dir(opt)
    suffix = _build_packet_loss_suffix(opt)

    # Save to CSV
    csv_path = os.path.join(output_dir, f'packet_loss_robustness{suffix}.csv')
    df_results.to_csv(csv_path, index=False)

    print("\n" + "="*80)
    print("PACKET LOSS ROBUSTNESS TEST RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to: {csv_path}")

    # Also save separate CSV for each range (for easier plotting)
    for range_name in ['short', 'middle', 'long']:
        df_range = df_results[['compression_ratio', 'packet_loss_pct',
                               f'AP@0.3_{range_name}', f'AP@0.5_{range_name}', f'AP@0.7_{range_name}']].copy()
        df_range.columns = ['compression_ratio', 'packet_loss_pct', 'AP@0.3', 'AP@0.5', 'AP@0.7']
        range_csv_path = os.path.join(output_dir, f'packet_loss_robustness_{range_name}{suffix}.csv')
        df_range.to_csv(range_csv_path, index=False)
        print(f"Range-specific results saved to: {range_csv_path}")

    # Also save method name and test info
    info_path = os.path.join(output_dir, f'packet_loss_test_info{suffix}.txt')
    with open(info_path, 'w') as f:
        f.write(f"Model Directory: {opt.model_dir}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Fusion Method: {opt.fusion_method}\n")
        f.write(f"Random Seed: {opt.seed}\n")
        f.write(f"Test Ratios: {opt.test_ratios}\n")
        f.write(f"Packet Loss Mode: {opt.packet_loss_mode}\n")
        f.write(f"Burst Coarse Shape: ({int(opt.burst_coarse_h)}, {int(opt.burst_coarse_w)})\n")
        f.write(f"Packet Loss Seed Base: {opt.packet_loss_seed_base}\n")
        f.write(f"Note: {opt.note}\n")
        f.write(f"\nPer-ratio speed metrics are saved in packet_loss_robustness{suffix}.csv\n")
        f.write(f"\nDistance Ranges:\n")
        f.write(f"  Short: 0-30m\n")
        f.write(f"  Middle: 30-50m\n")
        f.write(f"  Long: 50-100m\n")

    print(f"Test info saved to: {info_path}\n")


if __name__ == '__main__':
    main()
