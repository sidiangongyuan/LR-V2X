# Based on OpenCOOD inference utilities.
# License: TDG-Attribution-NonCommercial-NoDistrib
"""Evaluate LiDAR collaborative detection with random or spatial-burst packet loss."""

import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from opencood.data_utils import SUPER_CLASS_MAP
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import inference_utils, inference_utils_mc, train_utils
from opencood.utils import eval_utils, eval_utils_mc


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_packet_loss(hypes: dict, retention: float, opt: argparse.Namespace) -> dict:
    """Set the received fraction, not the pre-loss message size."""
    if not 0 <= retention <= 1:
        raise ValueError('Retention must be between 0 and 1.')
    hypes = copy.deepcopy(hypes)
    args = hypes['model']['args']
    overrides = {
        'compression_ratio': retention,
        'packet_loss_mode': opt.packet_loss_mode,
        'burst_coarse_h': opt.burst_coarse_h,
        'burst_coarse_w': opt.burst_coarse_w,
        'packet_loss_seed_base': opt.mask_seed,
    }
    args.update(overrides)
    for value in args.values():
        if isinstance(value, dict):
            value.update(overrides)
    return hypes


def empty_stats(with_scores: bool = True) -> dict:
    stats = {iou: {'tp': [], 'fp': [], 'gt': 0} for iou in (0.3, 0.5, 0.7)}
    if with_scores:
        for item in stats.values():
            item['score'] = []
    return stats


def update_class_stats(stats: dict, boxes, scores, gt_boxes, gt_labels) -> None:
    for class_id, class_name in enumerate(stats, start=1):
        gt = gt_boxes[gt_labels == class_id]
        if boxes is None or len(boxes) == 0:
            pred, confidence = None, None
        else:
            keep = scores[:, -1] == class_id
            pred, confidence = boxes[keep], scores[keep, 0]
        for iou in (0.3, 0.5, 0.7):
            eval_utils_mc.caluclate_tp_fp(pred, confidence, gt, stats[class_name], iou)


def evaluate(model, dataset, hypes: dict, opt: argparse.Namespace, device) -> dict:
    multi_class = hypes['fusion']['dataset'] == 'v2xreal'
    if multi_class:
        stats = {name: empty_stats(False) for name in SUPER_CLASS_MAP}
    else:
        stats = {name: empty_stats() for name in ('overall', 'short', 'middle', 'long')}
    ranges = {'overall': (None, None), 'short': (0, 30),
              'middle': (30, 50), 'long': (50, 100)}
    loader = DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=dataset.collate_batch_test, pin_memory=False, drop_last=False,
    )
    elapsed, frames = 0.0, 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader)):
            if batch is None:
                continue
            if opt.mask_seed is not None:
                batch['ego']['sample_idx'] = index
            batch = train_utils.to_device(batch, device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if multi_class:
                boxes, scores, gt_boxes, gt_labels = (
                    inference_utils_mc.inference_intermediate_fusion(batch, model, dataset)
                )
            else:
                result = inference_utils.inference_intermediate_fusion(batch, model, dataset)
                boxes, scores, gt_boxes = (
                    result['pred_box_tensor'], result['pred_score'], result['gt_box_tensor']
                )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            frames += 1
            if multi_class:
                update_class_stats(stats, boxes, scores, gt_boxes, gt_labels)
            else:
                for name, (left, right) in ranges.items():
                    for iou in (0.3, 0.5, 0.7):
                        if left is None:
                            eval_utils.caluclate_tp_fp(boxes, scores, gt_boxes, stats[name], iou)
                        else:
                            eval_utils.caluclate_tp_fp(
                                boxes, scores, gt_boxes, stats[name], iou,
                                left_range=left, right_range=right,
                            )
            torch.cuda.empty_cache()
    if not frames:
        raise RuntimeError('No valid evaluation frames were found.')
    metrics = {'frames': frames, 'FPS': frames / elapsed, 'avg_time_ms': 1000 * elapsed / frames}
    calculator = eval_utils_mc.calculate_ap if multi_class else eval_utils.calculate_ap
    for name, group in stats.items():
        for iou in (0.3, 0.5, 0.7):
            metrics[f'{name}_AP@{iou}'] = calculator(group, iou)[0]
    if multi_class:
        for iou in (0.3, 0.5, 0.7):
            metrics[f'mAP@{iou}'] = np.mean([metrics[f'{name}_AP@{iou}'] for name in stats])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True, help='Directory with config.yaml and checkpoints')
    parser.add_argument('--retention', nargs='+', type=float, default=[0.1],
                        help='Received fraction; 0.1 means 90%% packet loss, 1 means no loss')
    parser.add_argument('--packet_loss_mode', choices=['bernoulli', 'burst'], default='bernoulli')
    parser.add_argument('--burst_coarse_h', type=int, default=8)
    parser.add_argument('--burst_coarse_w', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_seed', type=int, default=None,
                        help='Optional independent per-frame mask seed')
    parser.add_argument('--output', default=None, help='CSV path; defaults to model_dir/evaluation.csv')
    opt = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_all(opt.seed)
    hypes = yaml_utils.load_yaml(None, opt)
    multi_class = hypes['fusion']['dataset'] == 'v2xreal'
    if multi_class and opt.mask_seed is None:
        opt.mask_seed = opt.seed
    hypes['validate_dir'] = hypes['test_dir']
    dataset = build_dataset(hypes, visualize=False, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for retention in opt.retention:
        if multi_class:
            seed_all(opt.seed)
        config = configure_packet_loss(hypes, retention, opt)
        model = train_utils.create_model(config)
        epoch, model = train_utils.load_saved_model(opt.model_dir, model)
        if not epoch:
            raise FileNotFoundError(f'No checkpoint found in {opt.model_dir}')
        model = model.to(device).eval()
        row = {
            'retention': retention, 'packet_loss_pct': 100 * (1 - retention),
            'packet_loss_mode': opt.packet_loss_mode, 'seed': opt.seed,
            'mask_seed': opt.mask_seed, 'epoch': epoch,
        }
        row.update(evaluate(model, dataset, config, opt, device))
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    output = Path(opt.output or os.path.join(opt.model_dir, 'evaluation.csv'))
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
