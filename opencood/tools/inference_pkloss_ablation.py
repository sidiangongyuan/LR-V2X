# -*- coding: utf-8 -*-
# Purpose: Packet-loss ablation for DiffV2X reconstruction modules
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Packet loss robustness testing with component ablations.

This script extends `inference_pkloss.py` with two ablation controls:
  - prior_only: use the prior decoder output directly (skip diffusion refinement)
  - zero_prior: use a zero prior (no latent prior decoder) and run diffusion

It is intended for evaluating reconstruction components under communication packet loss.
Results are saved in CSV format (similar to `inference_pkloss.py`), with an extra `mode` column.
"""

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils

torch.multiprocessing.set_sharing_strategy("file_system")


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_parser():
    parser = argparse.ArgumentParser(description="Packet Loss Ablation Testing (prior-only / zero-prior)")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        help="Fusion method: no, late, early, or intermediate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--note", default="", type=str, help="Any note")

    parser.add_argument(
        "--test_ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Compression ratios to test (0.1 = 90%% packet loss)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        type=str,
        default=["prior_only", "zero_prior"],
        choices=["full", "prior_only", "zero_prior"],
        help="Ablation modes to evaluate",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional output directory (default: model_dir)",
    )

    opt = parser.parse_args()
    return opt


def _set_compression_ratio(hypes: dict, compression_ratio: float) -> int:
    """
    Set compression_ratio in common model config locations.
    Returns the number of locations updated.
    """
    set_count = 0

    if "model" in hypes and "args" in hypes["model"]:
        hypes["model"]["args"]["compression_ratio"] = compression_ratio
        set_count += 1

        for key, value in hypes["model"]["args"].items():
            if isinstance(value, dict):
                value["compression_ratio"] = compression_ratio
                set_count += 1

    return set_count


def _apply_ablation_mode(hypes: dict, mode: str) -> None:
    """
    Apply ablation settings in-place.

    Notes:
      - `prior_only` should preserve the latent prior decoder behavior while skipping diffusion refinement.
      - `zero_prior` disables latent/statistical prior and uses zero prior with diffusion.
    """
    if "model" not in hypes or "args" not in hypes["model"]:
        return

    model_args = hypes["model"]["args"]

    if mode == "full":
        # Keep config as-is.
        return

    if mode == "prior_only":
        model_args["use_prior_only"] = True
        if isinstance(model_args.get("diffusion"), dict):
            model_args["diffusion"]["use_zero_prior"] = False
            model_args["diffusion"]["use_statistical_prior"] = False
            model_args["diffusion"]["use_latent_prior"] = True
        return

    if mode == "zero_prior":
        model_args["use_prior_only"] = False
        if isinstance(model_args.get("diffusion"), dict):
            model_args["diffusion"]["use_zero_prior"] = True
            model_args["diffusion"]["use_latent_prior"] = False
            model_args["diffusion"]["use_statistical_prior"] = False
        return


def load_model(hypes: dict, saved_path: str, device: torch.device):
    model = train_utils.create_model(hypes)
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"Loading model from checkpoint at epoch {resume_epoch}")
    model = model.to(device)
    model.eval()
    return model


def inference_single_setting(opt, hypes_base: dict, mode: str, compression_ratio: float, device, dataset):
    print("\n" + "=" * 80)
    print(f"[Mode] {mode} | compression_ratio={compression_ratio:.2f} ({(1 - compression_ratio) * 100:.1f}% loss)")
    print("=" * 80)

    hypes = copy.deepcopy(hypes_base)
    set_count = _set_compression_ratio(hypes, compression_ratio)
    if set_count == 0:
        print("[Warning] compression_ratio not found in config; verify model supports packet loss control.")

    _apply_ablation_mode(hypes, mode)

    model = load_model(hypes, opt.model_dir, device)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    result_stat_short = copy.deepcopy(result_stat)
    result_stat_middle = copy.deepcopy(result_stat)
    result_stat_long = copy.deepcopy(result_stat)

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc=f"{mode}|{compression_ratio:.1f}"):
            if batch_data is None:
                continue

            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == "late":
                infer_result = inference_utils.inference_late_fusion(batch_data, model, dataset)
            elif opt.fusion_method == "early":
                infer_result = inference_utils.inference_early_fusion(batch_data, model, dataset)
            elif opt.fusion_method == "intermediate":
                infer_result = inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            elif opt.fusion_method == "no":
                infer_result = inference_utils.inference_no_fusion(batch_data, model, dataset)
            else:
                raise NotImplementedError(f"Fusion method {opt.fusion_method} not supported")

            pred_box_tensor = infer_result["pred_box_tensor"]
            gt_box_tensor = infer_result["gt_box_tensor"]
            pred_score = infer_result["pred_score"]

            for iou_threshold in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou_threshold)

                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat_short,
                    iou_threshold,
                    left_range=0,
                    right_range=30,
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat_middle,
                    iou_threshold,
                    left_range=30,
                    right_range=50,
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat_long,
                    iou_threshold,
                    left_range=50,
                    right_range=100,
                )

            torch.cuda.empty_cache()

    def calc_ap(stats, thr):
        return eval_utils.calculate_ap(stats, thr)[0]

    ap = {
        "AP@0.3": calc_ap(result_stat, 0.30),
        "AP@0.5": calc_ap(result_stat, 0.50),
        "AP@0.7": calc_ap(result_stat, 0.70),
        "AP@0.3_short": calc_ap(result_stat_short, 0.30),
        "AP@0.5_short": calc_ap(result_stat_short, 0.50),
        "AP@0.7_short": calc_ap(result_stat_short, 0.70),
        "AP@0.3_middle": calc_ap(result_stat_middle, 0.30),
        "AP@0.5_middle": calc_ap(result_stat_middle, 0.50),
        "AP@0.7_middle": calc_ap(result_stat_middle, 0.70),
        "AP@0.3_long": calc_ap(result_stat_long, 0.30),
        "AP@0.5_long": calc_ap(result_stat_long, 0.50),
        "AP@0.7_long": calc_ap(result_stat_long, 0.70),
    }

    del model
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "compression_ratio": compression_ratio,
        "packet_loss_pct": (1 - compression_ratio) * 100,
        **ap,
    }


def _format_and_save(df: pd.DataFrame, save_dir: str, basename: str) -> None:
    cols_order = [
        "mode",
        "compression_ratio",
        "packet_loss_pct",
        "AP@0.3",
        "AP@0.5",
        "AP@0.7",
        "AP@0.3_short",
        "AP@0.5_short",
        "AP@0.7_short",
        "AP@0.3_middle",
        "AP@0.5_middle",
        "AP@0.7_middle",
        "AP@0.3_long",
        "AP@0.5_long",
        "AP@0.7_long",
    ]
    df = df[cols_order].copy()

    df["compression_ratio"] = df["compression_ratio"].apply(lambda x: f"{x:.2f}")
    df["packet_loss_pct"] = df["packet_loss_pct"].apply(lambda x: f"{x:.1f}%")
    for col in [c for c in df.columns if c.startswith("AP@")]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    csv_path = os.path.join(save_dir, f"{basename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Save] {csv_path}")

    for range_name in ["short", "middle", "long"]:
        df_range = df[["mode", "compression_ratio", "packet_loss_pct",
                       f"AP@0.3_{range_name}", f"AP@0.5_{range_name}", f"AP@0.7_{range_name}"]].copy()
        df_range.columns = ["mode", "compression_ratio", "packet_loss_pct", "AP@0.3", "AP@0.5", "AP@0.7"]
        range_csv_path = os.path.join(save_dir, f"{basename}_{range_name}.csv")
        df_range.to_csv(range_csv_path, index=False)
        print(f"[Save] {range_csv_path}")


def main():
    opt = test_parser()
    seed_all(opt.seed)

    save_dir = opt.save_dir or opt.model_dir
    os.makedirs(save_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, opt)
    print("Dataset Building")
    dataset = build_dataset(hypes, visualize=False, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = []
    for mode in opt.modes:
        for ratio in opt.test_ratios:
            all_results.append(inference_single_setting(opt, hypes, mode, ratio, device, dataset))

    df = pd.DataFrame(all_results)
    basename = "packet_loss_ablation"
    if opt.note:
        safe_note = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in opt.note.strip())
        basename = f"{basename}_{safe_note}"

    _format_and_save(df, save_dir, basename)

    info_path = os.path.join(save_dir, f"{basename}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Model Directory: {opt.model_dir}\n")
        f.write(f"Save Directory: {save_dir}\n")
        f.write(f"Fusion Method: {opt.fusion_method}\n")
        f.write(f"Seed: {opt.seed}\n")
        f.write(f"Test Ratios: {opt.test_ratios}\n")
        f.write(f"Modes: {opt.modes}\n")
        f.write(f"Note: {opt.note}\n")

    print(f"[Save] {info_path}")


if __name__ == "__main__":
    main()

