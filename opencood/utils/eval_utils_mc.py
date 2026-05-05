# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch
import pandas as pd

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']
    if gt_total == 0:
        return 0.0, [0.0, 1.0], [0.0, 0.0]

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, fps_dict=None):
    dump_dict = {}
    for class_name in result_stat.keys():
        dump_dict[class_name] = {}
        for iou_threshold in result_stat[class_name].keys():
            ap, mrec, mpre = calculate_ap(result_stat[class_name], iou_threshold)
            dump_dict[class_name].update(
                                  {iou_threshold:
                                       {"ap": ap,
                                        "mrec": mrec,
                                        "mpre": mpre
                                        }
                                   })
            print(f'{class_name}: AP@{iou_threshold} is {ap:.4f}', end=' ')
        print("")

    class_names = list(result_stat.keys())
    iou_thresholds = list(result_stat[class_names[0]].keys())

    # Calculate and print mAP
    mAP_dict = {}
    for iou_threshold in iou_thresholds:
        mAP = 0
        for class_name in class_names:
            mAP += dump_dict[class_name][iou_threshold]['ap']
        mAP_dict[iou_threshold] = mAP / len(class_names)
        print(f'mAP@{iou_threshold} is {mAP_dict[iou_threshold]:.4f}', end=' ')
    print("")

    # Save detailed results to yaml (keep for backward compatibility)
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))

    # ============================================================
    # NEW: Save results as CSV table for easy copying to papers
    # ============================================================

    # Prepare data for DataFrame
    # Columns: Method | AP_vehicle@0.3 | AP_vehicle@0.5 | AP_vehicle@0.7 |
    #                 | AP_ped@0.3 | AP_ped@0.5 | AP_ped@0.7 |
    #                 | AP_truck@0.3 | AP_truck@0.5 | AP_truck@0.7 |
    #                 | mAP@0.3 | mAP@0.5 | mAP@0.7 | FPS (Inference) | FPS (End-to-End)

    results_row = {}

    # Extract method name from save_path
    method_name = os.path.basename(save_path.rstrip('/'))
    results_row['Method'] = method_name

    # Add per-class APs
    for class_name in sorted(class_names):  # Sort for consistent order
        for iou_threshold in sorted(iou_thresholds):
            col_name = f'AP_{class_name}@{iou_threshold}'
            ap_value = dump_dict[class_name][iou_threshold]['ap']
            results_row[col_name] = f'{ap_value:.4f}'

    # Add mAPs
    for iou_threshold in sorted(iou_thresholds):
        col_name = f'mAP@{iou_threshold}'
        results_row[col_name] = f'{mAP_dict[iou_threshold]:.4f}'

    # Add FPS metrics if provided
    if fps_dict is not None:
        if 'inference_only' in fps_dict:
            results_row['FPS (Inference)'] = f"{fps_dict['inference_only']:.2f}"
        if 'end_to_end' in fps_dict:
            results_row['FPS (End-to-End)'] = f"{fps_dict['end_to_end']:.2f}"
        if 'avg_time_ms' in fps_dict:
            results_row['Avg Time (ms)'] = f"{fps_dict['avg_time_ms']:.2f}"

    # Create DataFrame
    df = pd.DataFrame([results_row])

    # Save to CSV
    csv_path = os.path.join(save_path, 'results_table.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n[Results] Saved table to: {csv_path}')

    # Print table to console
    print('\n' + '='*80)
    print('RESULTS TABLE')
    print('='*80)
    print(df.to_string(index=False))
    print('='*80 + '\n')
