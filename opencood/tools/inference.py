# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import sys
import argparse
import os
import random
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')




import atexit
from datetime import datetime

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
        # helps tqdm decide how to render
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

def redirect_print_to_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

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


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default="/mnt/sdc/public/data/yangk/result/quantv2x/dairv2x_diffv2x_stage2_20251208_154904/",
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,51.2",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--seed', default=42, type=int, 
                        help='random seed for results reproduction')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--use_prior_only', action='store_true',
                        help='Ablation: use prior decoder output directly without diffusion refinement')
    parser.add_argument('--use_zero_prior', action='store_true',
                        help='Ablation: use zero prior (random noise) instead of latent prior')
    parser.add_argument('--inference_timestep', type=int, default=None,
                        help='Ablation: override inference timestep (e.g., 250, 500, 750, 1000)')
    parser.add_argument('--no_ego_feature', action='store_true',
                        help='Ablation: disable ego feature in diffusion condition')
    parser.add_argument('--no_transform', action='store_true',
                        help='Ablation: disable transform matrix in diffusion condition')
    parser.add_argument('--no_latent', action='store_true',
                        help='Ablation: disable latent feature in diffusion condition')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    seed_all(opt.seed)
    redirect_print_to_log(opt.model_dir) 
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    # Ablation: Override use_prior_only if specified via command line
    if opt.use_prior_only:
        print(f"\n{'='*80}")
        print(f"[ABLATION] Overriding config: use_prior_only = True")
        print(f"[ABLATION] This will skip diffusion refinement and use prior directly")
        print(f"{'='*80}\n")
        if 'model' in hypes and 'args' in hypes['model']:
            hypes['model']['args']['use_prior_only'] = True
            opt.note += "_prior_only"

    # Ablation: Override use_zero_prior if specified via command line
    if opt.use_zero_prior:
        print(f"\n{'='*80}")
        print(f"[ABLATION] Overriding config: use_zero_prior = True")
        print(f"[ABLATION] This will use zero prior (random noise) instead of latent prior")
        print(f"{'='*80}\n")
        if 'model' in hypes and 'args' in hypes['model']:
            if 'diffusion' in hypes['model']['args']:
                hypes['model']['args']['diffusion']['use_zero_prior'] = True
                hypes['model']['args']['diffusion']['use_latent_prior'] = False
                hypes['model']['args']['diffusion']['use_statistical_prior'] = False
            opt.note += "_zero_prior"

    # Ablation: Override inference_timestep if specified via command line
    if opt.inference_timestep is not None:
        print(f"\n{'='*80}")
        print(f"[ABLATION] Overriding config: inference_timestep = {opt.inference_timestep}")
        print(f"[ABLATION] Testing different denoising timesteps")
        print(f"{'='*80}\n")
        if 'model' in hypes and 'args' in hypes['model']:
            if 'diffusion' in hypes['model']['args']:
                hypes['model']['args']['diffusion']['inference_timestep'] = opt.inference_timestep
            opt.note += f"_t{opt.inference_timestep}"

    # Ablation: Override condition components if specified via command line
    if opt.no_ego_feature or opt.no_transform or opt.no_latent:
        print(f"\n{'='*80}")
        print(f"[ABLATION] Overriding condition components:")
        print(f"  - Ego feature: {not opt.no_ego_feature}")
        print(f"  - Transform matrix: {not opt.no_transform}")
        print(f"  - Latent feature: {not opt.no_latent}")
        print(f"{'='*80}\n")
        if 'model' in hypes and 'args' in hypes['model']:
            if 'diffusion' in hypes['model']['args']:
                if opt.no_ego_feature:
                    hypes['model']['args']['diffusion']['use_ego_feature'] = False
                    opt.note += "_noego"
                if opt.no_transform:
                    hypes['model']['args']['diffusion']['use_transform'] = False
                    opt.note += "_notrans"
                if opt.no_latent:
                    hypes['model']['args']['diffusion']['use_latent'] = False
                    opt.note += "_nolat"

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir'] or "V2XREAL" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(700,800))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
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

    infer_info = opt.fusion_method + opt.note

    # Initialize timing variables for FPS calculation
    total_time = 0
    num_batches = 0
    import time


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            # Start timing for inference
            start_time = time.time()

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            # End timing for inference
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            num_batches += 1

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            for iou_threshold in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        iou_threshold)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_short,
                                        iou_threshold, 
                                        left_range=0,
                                        right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_middle,
                                        iou_threshold, 
                                        left_range=30,
                                        right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_long,
                                        iou_threshold,
                                        left_range=50,
                                        right_range=100)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                
                # vis_feat_save_path = os.path.join(opt.model_dir, f'feat_vis_{infer_info}')
                # vis_utils.visualize_feature_distribution(infer_result, vis_feat_save_path, i)


        torch.cuda.empty_cache()

    # ============================================================
    # FPS Calculation
    # ============================================================
    print("\n" + "="*60)
    print("INFERENCE SPEED METRICS")
    print("="*60)

    fps_dict = {}

    if num_batches > 0 and total_time > 0:
        inference_only_fps = num_batches / total_time
        avg_inference_time = total_time / num_batches * 1000  # ms
        fps_dict['inference_only'] = inference_only_fps
        fps_dict['avg_time_ms'] = avg_inference_time
        print(f"Inference-Only FPS: {inference_only_fps:.2f} frames/s")
        print(f"  (model forward pass only)")
        print(f"Average inference time per frame: {avg_inference_time:.2f} ms")
    else:
        print("No batches processed or total time is zero.")

    print("="*60 + "\n")

    # Evaluate results for different ranges with FPS info
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir, infer_info="short", fps_dict=fps_dict)
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir, infer_info="middle", fps_dict=fps_dict)
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir, infer_info="long", fps_dict=fps_dict)
    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info, fps_dict=fps_dict)

if __name__ == '__main__':
    main()
