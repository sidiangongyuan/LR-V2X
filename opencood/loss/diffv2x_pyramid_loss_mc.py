# -*- coding: utf-8 -*-
"""
DiffV2X Pyramid Loss for Multi-Class Detection

Integrates:
- Pyramid fusion loss (from PointPillarPyramidLossMC)
- Diffusion reconstruction loss
"""

import torch
import torch.nn as nn
from opencood.loss.point_pillar_pyramid_loss_mc import PointPillarPyramidLossMC


class DiffV2XPyramidLossMC(PointPillarPyramidLossMC):
    """
    DiffV2X Pyramid Loss for Multi-Class Detection

    Combines:
    - Detection losses (cls, reg, dir, pyramid) from parent class
    - Diffusion reconstruction loss
    """

    def __init__(self, args):
        super().__init__(args)

        # Diffusion loss configuration
        self.diffusion_weight = args.get('diffusion', {}).get('weight', 0.0)

        print(f"\n{'='*80}")
        print(f"DiffV2X Pyramid Loss Initialization")
        print(f"{'='*80}")
        print(f"Diffusion loss weight: {self.diffusion_weight}")
        print(f"{'='*80}\n")

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Forward pass for DiffV2X loss calculation.

        Parameters
        ----------
        output_dict : dict
            Dictionary containing model outputs:
            - 'cls_preds': Classification predictions
            - 'reg_preds': Regression predictions
            - 'dir_preds': Direction predictions
            - 'diffusion_loss': Diffusion reconstruction loss (optional)
            - 'occ_single_list': Pyramid occupancy predictions

        target_dict : dict
            Dictionary containing ground truth labels

        suffix : str
            Suffix for loss names (e.g., '_single' for single agent supervision)

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss value
        """

        # Calculate detection losses (from parent class)
        if output_dict['pyramid'] == 'collab':
            detection_loss = self.forward_collab(output_dict, target_dict, suffix)
        elif output_dict['pyramid'] == 'single':
            detection_loss = self.forward_single(output_dict, target_dict, suffix)
        else:
            raise ValueError(f"Unknown pyramid mode: {output_dict['pyramid']}")

        # Add diffusion loss if present
        diffusion_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=detection_loss.device))

        # Weighted sum
        total_loss = detection_loss + self.diffusion_weight * diffusion_loss

        # Update loss dict
        self.loss_dict.update({
            'diffusion_loss': diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss,
            'total_loss': total_loss.item()
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None, suffix=""):
        """
        Print out the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training.
        writer : SummaryWriter
            Used to visualize on tensorboard
        pbar : tqdm.pbar
            Progress bar
        suffix : str
            Suffix for loss names
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)
        diffusion_loss = self.loss_dict.get('diffusion_loss', 0)

        if writer is not None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                            epoch * batch_len + batch_id)
            writer.add_scalar('Diffusion_loss' + suffix, diffusion_loss,
                            epoch * batch_len + batch_id)

        if pbar is None:
            print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf: %.4f"
                  " || Loc: %.4f || Dir: %.4f || IoU: %.4f || Depth: %.4f"
                  " || Pyramid: %.4f || Diff: %.4f" % (
                      epoch, batch_id + 1, batch_len, suffix,
                      total_loss, cls_loss, reg_loss, dir_loss, iou_loss,
                      depth_loss, pyramid_loss, diffusion_loss))
        else:
            pbar.set_description("[epoch %d][%d/%d]%s || Loss: %.4f || Conf: %.4f"
                                " || Loc: %.4f || Dir: %.4f || IoU: %.4f || Depth: %.4f"
                                " || Pyramid: %.4f || Diff: %.4f" % (
                                    epoch, batch_id + 1, batch_len, suffix,
                                    total_loss, cls_loss, reg_loss, dir_loss, iou_loss,
                                    depth_loss, pyramid_loss, diffusion_loss))
