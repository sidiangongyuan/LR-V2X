# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
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


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    left_range=-float('inf'), right_range=float('inf')):
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
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
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

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove the bbx out of range
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score_new)

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
        
        result_stat[iou_thresh]['score'] += det_score_new.tolist()

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

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

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


def eval_final_results(result_stat, save_path, infer_info=None, fps_dict=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })

    # Save detailed yaml (keep for backward compatibility)
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    print(infer_info,
          'The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70))

    # ============================================================
    # NEW: Save results as CSV table for easy copying to papers
    # ============================================================

    # Try to read existing CSV to append new results
    csv_path = os.path.join(save_path, 'results_table.csv')

    # Prepare current result row
    results_row = {
        'Range': infer_info if infer_info else 'Overall',
        'AP@0.3': f'{ap_30:.4f}',
        'AP@0.5': f'{ap_50:.4f}',
        'AP@0.7': f'{ap_70:.4f}'
    }

    # Add FPS metrics if provided (only for the first range to avoid duplication)
    if fps_dict is not None and infer_info == 'short':
        if 'inference_only' in fps_dict:
            results_row['FPS (Inference)'] = f"{fps_dict['inference_only']:.2f}"
        if 'avg_time_ms' in fps_dict:
            results_row['Avg Time (ms)'] = f"{fps_dict['avg_time_ms']:.2f}"

    # Check if CSV already exists to append
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Check if this range already exists, if yes, update it
        if 'Range' in df_existing.columns:
            mask = df_existing['Range'] == results_row['Range']
            if mask.any():
                # Update existing row
                for col in results_row.keys():
                    if col in df_existing.columns:
                        df_existing.loc[mask, col] = results_row[col]
                    else:
                        df_existing[col] = ''
                        df_existing.loc[mask, col] = results_row[col]
                df = df_existing
            else:
                # Append new row
                df = pd.concat([df_existing, pd.DataFrame([results_row])], ignore_index=True)
        else:
            df = pd.concat([df_existing, pd.DataFrame([results_row])], ignore_index=True)
    else:
        # Create new DataFrame
        df = pd.DataFrame([results_row])

    # Save to CSV
    df.to_csv(csv_path, index=False)

    # If this is the final result (infer_info without range suffix), print the complete table
    if infer_info and infer_info not in ['short', 'middle', 'long']:
        print(f'\n[Results] Saved table to: {csv_path}')
        print('\n' + '='*60)
        print('RESULTS TABLE (All Ranges)')
        print('='*60)
        print(df.to_string(index=False))
        print('='*60 + '\n')

    return ap_30, ap_50, ap_70
