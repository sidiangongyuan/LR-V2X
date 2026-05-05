# -*- coding: utf-8 -*-
# Author: Based on inference_mc.py and inference_pkloss.py
# Purpose: Test multi-class detection under communication packet loss
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Multi-class inference script to test model robustness under communication packet loss.
Tests multiple compression ratios: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
Saves results to CSV tables for easy comparison across different classes.
"""

import sys
import argparse
import os

# Allow running this file directly without installing the package
# (e.g., without `pip install -e .`).
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
import atexit
from datetime import datetime

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils_mc, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils_mc
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class Tee:
    """Write to multiple streams (e.g., console + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def redirect_print_to_log(log_dir):
    """Redirect all prints to log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"inference_pkloss_mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_f)
    sys.stderr = Tee(old_err, log_f)

    def _cleanup():
        try:
            sys.stdout = old_out
            sys.stderr = old_err
        finally:
            try:
                log_f.close()
            except Exception:
                pass

    atexit.register(_cleanup)
    print(f"[LOG] Saving all prints to: {log_path}")
    return log_path


def test_parser():
    parser = argparse.ArgumentParser(description="Multi-class Packet Loss Robustness Testing")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional directory to save evaluation outputs. Defaults to model_dir.')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='Fusion method: late, early, intermediate, or nofusion')
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
    parser.add_argument('--dataset_mode', type=str, default="",
                        help='Dataset mode override')
    parser.add_argument('--epoch', default=None,
                        help='Epoch number to load model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='List of random seeds for repeated evaluation. '
                             'Overrides --seed if provided.')
    parser.add_argument('--inference_timestep', type=int, default=None,
                        help='Override diffusion inference timestep (e.g., 150, 250, 500).')

    # Packet loss ratios to test
    parser.add_argument('--test_ratios', nargs='+', type=float,
                        default=[0.3, 0.5, 0.7, 0.9, 1],
                        help='Compression ratios to test (0.1 = 90%% packet loss)')

    opt = parser.parse_args()
    return opt


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


def _get_output_dir(opt: argparse.Namespace) -> str:
    output_dir = opt.save_dir if opt.save_dir else opt.model_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _normalize_sample_indices(sample_indices, dataset_len: int):
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


def override_inference_timestep(hypes: dict, inference_timestep: int) -> int:
    """
    Override diffusion inference_timestep in config (best-effort).

    Returns:
        set_count (int): number of locations updated.
    """
    set_count = 0
    if not isinstance(hypes, dict):
        return set_count
    model_args = hypes.get('model', {}).get('args', None)
    if not isinstance(model_args, dict):
        return set_count

    handled_dict_ids = set()

    # Common location used by DiffV2X models.
    diffusion_cfg = model_args.get('diffusion', None)
    if isinstance(diffusion_cfg, dict):
        diffusion_cfg['inference_timestep'] = int(inference_timestep)
        handled_dict_ids.add(id(diffusion_cfg))
        set_count += 1

    # Also update any nested dicts that explicitly contain `inference_timestep`.
    def _traverse(obj):
        nonlocal set_count
        if isinstance(obj, dict):
            if 'inference_timestep' in obj and id(obj) not in handled_dict_ids:
                obj['inference_timestep'] = int(inference_timestep)
                set_count += 1
            for v in obj.values():
                _traverse(v)
        elif isinstance(obj, list):
            for v in obj:
                _traverse(v)

    _traverse(model_args)
    return set_count


def load_model_with_compression_ratio(hypes, compression_ratio, saved_path, device):
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
    set_count = 0

    # 1. ALWAYS set at top-level model.args (for models that read from top-level)
    hypes['model']['args']['compression_ratio'] = compression_ratio
    print(f"[Config] Set compression_ratio={compression_ratio:.2f} at model.args (top-level)")
    set_count += 1

    # 2. Traverse all sub-modules in model.args and set compression_ratio
    if 'args' in hypes['model']:
        for key, value in hypes['model']['args'].items():
            if isinstance(value, dict):
                value['compression_ratio'] = compression_ratio
                print(f"[Config] Set compression_ratio={compression_ratio:.2f} at model.args.{key}")
                set_count += 1

    if set_count == 0:
        print(f"[Warning] No compression_ratio parameter found in model config.")
        print(f"[Warning] Make sure your model supports compression_ratio parameter!")
    else:
        print(f"[Config] Total: Set compression_ratio in {set_count} location(s)")

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
    Run multi-class inference for a single compression ratio.

    Returns:
        results_dict: Dictionary with AP for all classes and IoU thresholds
    """
    print("\n" + "="*80)
    packet_loss_pct = (1 - compression_ratio) * 100
    print(
        f"Testing Compression Ratio: {compression_ratio:.2f} "
        f"({packet_loss_pct:.1f}% packet loss)"
    )
    print("="*80)

    # Load model with this compression ratio
    model = load_model_with_compression_ratio(hypes, compression_ratio, opt.model_dir, device)

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

    # Create result statistics for each class
    result_stat = {}
    for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys():
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = {'tp': [], 'fp': [], 'gt': 0}

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
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = (
                    inference_utils_mc.inference_late_fusion(
                        batch_data, model, opencood_dataset
                    )
                )
            elif opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = (
                    inference_utils_mc.inference_nofusion(
                        batch_data, model, opencood_dataset
                    )
                )
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = (
                    inference_utils_mc.inference_early_fusion(
                        batch_data, model, opencood_dataset
                    )
                )
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = (
                    inference_utils_mc.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset
                    )
                )
            else:
                raise NotImplementedError(
                    'Only early, late, intermediate and nofusion fusion is supported.'
                )
            _synchronize_if_needed(device)
            batch_time = time.time() - start_time
            total_time += batch_time
            num_batches += 1

            # Calculate TP/FP for each class
            # Handle case where pred_box_tensor is None or empty
            if pred_box_tensor is None or len(pred_box_tensor) == 0:
                # No predictions - only count ground truth
                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    for iou_threshold in result_stat[class_name].keys():
                        keep_index_gt = gt_label_tensor == class_id
                        eval_utils_mc.caluclate_tp_fp(None,
                                                      None,
                                                      gt_box_tensor[keep_index_gt, ...],
                                                      result_stat[class_name],
                                                      iou_threshold)
            else:
                # Normal case - predictions exist
                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    for iou_threshold in result_stat[class_name].keys():
                        keep_index_pred = pred_score[:, -1] == class_id
                        keep_index_gt = gt_label_tensor == class_id
                        eval_utils_mc.caluclate_tp_fp(pred_box_tensor[keep_index_pred, ...],
                                                      pred_score[keep_index_pred, 0],
                                                      gt_box_tensor[keep_index_gt, ...],
                                                      result_stat[class_name],
                                                      iou_threshold)

            # Optional visualization
            should_save_vis = False
            if opt.save_vis:
                if dataset_index_lookup is not None:
                    should_save_vis = True
                else:
                    should_save_vis = (i % opt.save_vis_interval == 0)

            if should_save_vis:
                vis_dir = opt.vis_tag if opt.vis_tag else f'vis_pkloss_{compression_ratio:.1f}'
                if opt.inference_timestep is not None:
                    vis_dir += f'_t{opt.inference_timestep}'
                vis_save_path = os.path.join(_get_output_dir(opt), vis_dir)
                os.makedirs(vis_save_path, exist_ok=True)
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % dataset_index)
                simple_vis.visualize(
                    {
                        'pred_box_tensor': pred_box_tensor,
                        'gt_box_tensor': gt_box_tensor,
                        'score_tensor': pred_score
                    },
                    batch_data['ego']['origin_lidar'][0],
                    hypes['postprocess']['gt_range'],
                    vis_save_path,
                    method='bev',
                    left_hand=True
                )

            torch.cuda.empty_cache()

    # Calculate final metrics for each class
    results = {
        'compression_ratio': compression_ratio,
        'packet_loss_pct': (1 - compression_ratio) * 100,
    }

    print(f"\nResults for compression_ratio={compression_ratio:.2f}:")
    for class_name in result_stat.keys():
        ap_30, _, _ = eval_utils_mc.calculate_ap(result_stat[class_name], 0.30)
        ap_50, _, _ = eval_utils_mc.calculate_ap(result_stat[class_name], 0.50)
        ap_70, _, _ = eval_utils_mc.calculate_ap(result_stat[class_name], 0.70)

        print(f"  {class_name:15s} - AP@0.3: {ap_30:.4f}, AP@0.5: {ap_50:.4f}, AP@0.7: {ap_70:.4f}")

        results[f'{class_name}_AP@0.3'] = ap_30
        results[f'{class_name}_AP@0.5'] = ap_50
        results[f'{class_name}_AP@0.7'] = ap_70

    # Mean AP across super-classes (simple macro average)
    class_names = list(result_stat.keys())
    for iou in [0.3, 0.5, 0.7]:
        key = f"mAP@{iou}"
        results[key] = float(np.mean([results[f"{c}_AP@{iou}"] for c in class_names]))

    if num_batches > 0 and total_time > 0:
        results['FPS (Inference)'] = float(num_batches / total_time)
        results['Avg Time (ms)'] = float(total_time / num_batches * 1000.0)
        print(f"  FPS             - Inference-only: {results['FPS (Inference)']:.2f} frames/s")
        print(f"  Avg Time (ms)   - {results['Avg Time (ms)']:.2f}")
    else:
        results['FPS (Inference)'] = float('nan')
        results['Avg Time (ms)'] = float('nan')
        print("  FPS             - Inference-only: N/A")
        print("  Avg Time (ms)   - N/A")

    # Clean up model
    del model
    torch.cuda.empty_cache()

    return results


def main():
    opt = test_parser()

    # Redirect prints to log file
    output_dir = _get_output_dir(opt)
    redirect_print_to_log(output_dir)

    # Assertions
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion'], \
        'Fusion method must be one of: late, early, intermediate, nofusion'

    seeds = opt.seeds if opt.seeds else [opt.seed]
    print(f"Seeds: {seeds}")

    # Load configuration
    hypes = yaml_utils.load_yaml(None, opt)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode

    # Optional override: diffusion inference_timestep (global, fixed).
    if opt.inference_timestep is not None:
        set_count = override_inference_timestep(hypes, opt.inference_timestep)
        if set_count == 0:
            print("[Warning] --inference_timestep is set but config has no `inference_timestep`.")
            print("[Warning] Model may ignore it (filenames will still be suffixed).")
        else:
            print(f"[Config] Overrode inference_timestep = {opt.inference_timestep} "
                  f"in {set_count} location(s).")

        diffusion_cfg = hypes.get('model', {}).get('args', {}).get('diffusion', {})
        if isinstance(diffusion_cfg, dict) and 'num_timesteps' in diffusion_cfg:
            num_timesteps = diffusion_cfg.get('num_timesteps')
            if isinstance(num_timesteps, int) and num_timesteps > 0:
                assert 0 <= int(opt.inference_timestep) < num_timesteps, (
                    f"inference_timestep={opt.inference_timestep} must be in "
                    f"[0, {num_timesteps - 1}] (num_timesteps={num_timesteps})"
                )

    print(f"Dataset mode: {hypes['dataset_mode']}")
    hypes['validate_dir'] = hypes['test_dir']  # change to test_dir for inference

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    print(f"{len(opencood_dataset)} samples found.")
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
    all_results_mean = []
    all_results_std = []
    all_results_raw = []

    for compression_ratio in opt.test_ratios:
        results_per_seed = []
        for seed in seeds:
            seed_all(seed)
            print(f"Random seed set to: {seed}")
            result = inference_single_ratio(
                opt,
                hypes,
                compression_ratio,
                device,
                opencood_dataset,
                eval_dataset=eval_dataset,
                dataset_index_lookup=dataset_index_lookup,
            )
            result['seed'] = seed
            results_per_seed.append(result)

        df_ratio = pd.DataFrame(results_per_seed)
        all_results_raw.append(df_ratio)

        meta_cols = ['compression_ratio', 'packet_loss_pct', 'seed']
        metric_cols = [c for c in df_ratio.columns if c not in meta_cols]
        df_metrics = df_ratio[metric_cols].astype(float)

        mean_row = {
            'compression_ratio': float(compression_ratio),
            'packet_loss_pct': float((1 - compression_ratio) * 100),
            'num_seeds': int(len(seeds)),
        }
        std_row = {
            'compression_ratio': float(compression_ratio),
            'packet_loss_pct': float((1 - compression_ratio) * 100),
            'num_seeds': int(len(seeds)),
        }
        for col in metric_cols:
            mean_row[col] = float(df_metrics[col].mean())
            std_row[col] = float(df_metrics[col].std(ddof=0))

        all_results_mean.append(mean_row)
        all_results_std.append(std_row)

    # ============================================================
    # Save results to CSV
    # ============================================================
    df_results_mean = pd.DataFrame(all_results_mean)
    df_results_std = pd.DataFrame(all_results_std)
    if all_results_raw:
        df_results_raw = pd.concat(all_results_raw, ignore_index=True)
    else:
        df_results_raw = pd.DataFrame()

    # Reorder columns: compression_ratio, packet_loss_pct, then all class results
    base_cols = ['compression_ratio', 'packet_loss_pct', 'num_seeds']
    metric_cols = [col for col in df_results_mean.columns if col not in base_cols]
    df_results_mean = df_results_mean[base_cols + metric_cols]
    df_results_std = df_results_std[base_cols + metric_cols]

    # Format for better readability
    df_results_display = df_results_mean.copy()
    df_results_display['compression_ratio'] = df_results_display['compression_ratio'].apply(
        lambda x: f'{x:.2f}'
    )
    df_results_display['packet_loss_pct'] = df_results_display['packet_loss_pct'].apply(
        lambda x: f'{x:.1f}%'
    )

    # Format all AP columns
    ap_cols = [
        col
        for col in df_results_display.columns
        if ('AP@' in col) or col.startswith('mAP@')
    ]
    for col in ap_cols:
        df_results_display[col] = df_results_display[col].apply(lambda x: f'{x:.4f}')
    for col in ['FPS (Inference)', 'Avg Time (ms)']:
        if col in df_results_display.columns:
            df_results_display[col] = df_results_display[col].apply(
                lambda x: '' if pd.isna(x) else f'{x:.2f}'
            )

    t_suffix = f"_t{opt.inference_timestep}" if opt.inference_timestep is not None else ""

    # Save to CSV (with raw numbers for easier processing)
    csv_path = os.path.join(output_dir, f'packet_loss_robustness_mc{t_suffix}.csv')
    df_results_mean.to_csv(csv_path, index=False)
    csv_std_path = os.path.join(output_dir, f'packet_loss_robustness_mc_std{t_suffix}.csv')
    df_results_std.to_csv(csv_std_path, index=False)
    csv_raw_path = os.path.join(output_dir, f'packet_loss_robustness_mc_raw{t_suffix}.csv')
    df_results_raw.to_csv(csv_raw_path, index=False)

    print("\n" + "="*80)
    print("MULTI-CLASS PACKET LOSS ROBUSTNESS TEST RESULTS")
    print("="*80)
    print(df_results_display.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to: {csv_path}")
    print(f"Std results saved to: {csv_std_path}")
    print(f"Raw per-seed results saved to: {csv_raw_path}")

    # Also save separate CSV for each class (for easier plotting)
    class_names = list(opencood.data_utils.SUPER_CLASS_MAP.keys())
    for class_name in class_names:
        class_cols_select = ['compression_ratio', 'packet_loss_pct', 'num_seeds',
                             f'{class_name}_AP@0.3', f'{class_name}_AP@0.5', f'{class_name}_AP@0.7']

        df_class_mean = df_results_mean[class_cols_select].copy()
        df_class_std = df_results_std[class_cols_select].copy()
        df_class_mean.columns = [
            'compression_ratio', 'packet_loss_pct', 'num_seeds', 'AP@0.3', 'AP@0.5', 'AP@0.7'
        ]
        df_class_std.columns = [
            'compression_ratio', 'packet_loss_pct', 'num_seeds', 'AP@0.3', 'AP@0.5', 'AP@0.7'
        ]

        class_csv_path = os.path.join(
            output_dir, f'packet_loss_robustness_{class_name}{t_suffix}.csv'
        )
        df_class_mean.to_csv(class_csv_path, index=False)
        class_csv_std_path = os.path.join(
            output_dir, f'packet_loss_robustness_{class_name}_std{t_suffix}.csv'
        )
        df_class_std.to_csv(class_csv_std_path, index=False)
        print(f"Class-specific results saved to: {class_csv_path}")
        print(f"Class-specific std saved to: {class_csv_std_path}")

    # Save test info
    info_path = os.path.join(output_dir, f'packet_loss_test_info_mc{t_suffix}.txt')
    with open(info_path, 'w') as f:
        f.write(f"Model Directory: {opt.model_dir}\n")
        f.write(f"Fusion Method: {opt.fusion_method}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Test Ratios: {opt.test_ratios}\n")
        f.write(f"Dataset Mode: {hypes['dataset_mode']}\n")
        f.write(f"Inference Timestep: {opt.inference_timestep}\n")
        f.write(f"Per-ratio speed metrics are saved in packet_loss_robustness_mc{t_suffix}.csv\n")
        f.write(f"\nClasses Evaluated:\n")
        for class_name in class_names:
            f.write(f"  - {class_name}\n")

    print(f"Test info saved to: {info_path}\n")


if __name__ == '__main__':
    main()
