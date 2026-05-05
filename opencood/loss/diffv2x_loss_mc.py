"""
DiffV2X Loss Function for Multi-Class Detection

Combines:
1. Diffusion loss (noise prediction MSE)
2. Detection losses (classification, regression, direction)
3. Optional depth supervision
4. Optional perceptual/semantic consistency loss

Author: Implementation based on HEAL framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_depth_loss_mc import PointPillarDepthLossMC


class DiffV2XLossMC(PointPillarDepthLossMC):
    """
    DiffV2X Multi-Class Loss

    Inherits detection losses from PointPillarDepthLossMC and adds:
    - Diffusion loss (L_diffusion)
    - Optional perceptual loss (L_perceptual)

    Total loss:
        L_total = w_diff * L_diffusion + w_det * L_detection + w_depth * L_depth

    where L_detection includes classification, regression, and direction losses.
    """

    def __init__(self, args):
        """
        Args:
            args: Dictionary with loss configuration
                - diffusion: dict with diffusion loss settings
                    - weight: weight for diffusion loss (default: 1.0)
                    - use_perceptual: whether to use perceptual loss (default: False)
                    - perceptual_weight: weight for perceptual loss (default: 0.1)
                - cls, reg, dir, iou, depth: detection loss settings (inherited)
        """
        super().__init__(args)

        # Diffusion loss configuration
        self.diffusion_config = args.get('diffusion', {})
        self.diffusion_weight = self.diffusion_config.get('weight', 1.0)

        # Perceptual loss configuration (optional)
        self.use_perceptual = self.diffusion_config.get('use_perceptual', False)
        self.perceptual_weight = self.diffusion_config.get('perceptual_weight', 0.1)

        if self.use_perceptual:
            # Simple perceptual loss using feature similarity
            # Could use VGG or other pretrained features for stronger perceptual loss
            self.perceptual_loss_func = nn.L1Loss()

        print(f"[DiffV2X Loss] Diffusion weight: {self.diffusion_weight}")
        if self.use_perceptual:
            print(f"[DiffV2X Loss] Perceptual weight: {self.perceptual_weight}")

    def compute_diffusion_loss(self, output_dict):
        """
        Extract diffusion loss from output_dict.

        The diffusion loss is computed in the model's forward pass during training.

        Args:
            output_dict: Model output dictionary containing 'diffusion_loss'

        Returns:
            diffusion_loss: Scalar tensor
        """
        if 'diffusion_loss' not in output_dict:
            # No diffusion loss (likely inference mode)
            return torch.tensor(0.0, device=next(iter(output_dict.values())).device)

        return output_dict['diffusion_loss']

    def compute_perceptual_loss(self, output_dict, target_dict):
        """
        Compute perceptual loss for semantic consistency.

        Measures similarity between generated features and target features.
        This encourages the diffusion model to preserve semantic information.

        Args:
            output_dict: Model output dictionary
            target_dict: Target dictionary

        Returns:
            perceptual_loss: Scalar tensor
        """
        if not self.use_perceptual:
            return torch.tensor(0.0, device=next(iter(output_dict.values())).device)

        # Get generated and target features
        # For now, use a simple feature-level L1 loss
        # In practice, could use intermediate features or pretrained embeddings

        # This is a placeholder - actual implementation depends on what features to compare
        # Option 1: Compare BEV features before detection head
        # Option 2: Compare intermediate diffusion features
        # Option 3: Use pretrained VGG/ResNet features

        # For now, return zero as this is optional
        return torch.tensor(0.0, device=next(iter(output_dict.values())).device)

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Compute total loss for DiffV2X.

        Args:
            output_dict: Model output dictionary containing:
                - cls_preds: [B, H, W, num_anchors * num_class^2]
                - reg_preds: [B, H, W, num_anchors * num_class * 7]
                - dir_preds: [B, H, W, num_anchors * num_class * 2]
                - diffusion_loss: scalar (during training)
                - (optional) depth_items: depth supervision items
            target_dict: Target dictionary
            suffix: Optional suffix for multi-stage training

        Returns:
            total_loss: Scalar tensor
        """
        # 1. Compute detection losses (cls, reg, dir, iou, depth)
        detection_loss = super().forward(output_dict, target_dict, suffix)

        # 2. Compute diffusion loss
        diffusion_loss = self.compute_diffusion_loss(output_dict)
        weighted_diffusion_loss = diffusion_loss * self.diffusion_weight

        # 3. Compute perceptual loss (optional)
        perceptual_loss = self.compute_perceptual_loss(output_dict, target_dict)
        weighted_perceptual_loss = perceptual_loss * self.perceptual_weight

        # 4. Total loss
        total_loss = detection_loss + weighted_diffusion_loss + weighted_perceptual_loss

        # Update loss dictionary
        self.loss_dict.update({
            'diffusion_loss': diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss,
            'weighted_diffusion_loss': weighted_diffusion_loss.item() if isinstance(weighted_diffusion_loss, torch.Tensor) else weighted_diffusion_loss,
        })

        if self.use_perceptual:
            self.loss_dict.update({
                'perceptual_loss': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
            })

        # Update total loss
        self.loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None, suffix=""):
        """
        Print and log losses.

        Args:
            epoch: Current epoch
            batch_id: Current batch ID
            batch_len: Total batches in epoch
            writer: TensorBoard writer
            pbar: Progress bar (tqdm)
            suffix: Optional suffix
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        diffusion_loss = self.loss_dict.get('diffusion_loss', 0)
        weighted_diffusion_loss = self.loss_dict.get('weighted_diffusion_loss', 0)
        perceptual_loss = self.loss_dict.get('perceptual_loss', 0)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Total_loss' + suffix, total_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Regression_loss' + suffix, reg_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Diffusion_loss' + suffix, diffusion_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Weighted_Diffusion_loss' + suffix, weighted_diffusion_loss, epoch * batch_len + batch_id)

            if self.use_perceptual:
                writer.add_scalar('Perceptual_loss' + suffix, perceptual_loss, epoch * batch_len + batch_id)

        # Console logging
        log_str = (
            f"[epoch {epoch}][{batch_id + 1}/{batch_len}]{suffix} || "
            f"Loss: {total_loss:.4f} || "
            f"Diff: {diffusion_loss:.4f} (w: {weighted_diffusion_loss:.4f}) || "
            f"Conf: {cls_loss:.4f} || "
            f"Loc: {reg_loss:.4f} || "
            f"Dir: {dir_loss:.4f} || "
            f"IoU: {iou_loss:.4f} || "
            f"Depth: {depth_loss:.4f}"
        )

        if self.use_perceptual:
            log_str += f" || Perc: {perceptual_loss:.4f}"

        if pbar is None:
            print(log_str)
        else:
            pbar.set_description(log_str)

    def get_loss_breakdown(self):
        """
        Get detailed breakdown of all loss components.

        Returns:
            dict: Dictionary with all loss components and their contributions
        """
        total_loss = self.loss_dict.get('total_loss', 0)

        breakdown = {
            'total': total_loss,
            'detection': {
                'classification': self.loss_dict.get('cls_loss', 0),
                'regression': self.loss_dict.get('reg_loss', 0),
                'direction': self.loss_dict.get('dir_loss', 0),
                'iou': self.loss_dict.get('iou_loss', 0),
                'depth': self.loss_dict.get('depth_loss', 0),
            },
            'diffusion': {
                'raw': self.loss_dict.get('diffusion_loss', 0),
                'weighted': self.loss_dict.get('weighted_diffusion_loss', 0),
                'weight': self.diffusion_weight,
            }
        }

        if self.use_perceptual:
            breakdown['perceptual'] = {
                'raw': self.loss_dict.get('perceptual_loss', 0),
                'weight': self.perceptual_weight,
            }

        # Calculate percentages
        if total_loss > 0:
            detection_sum = sum(breakdown['detection'].values())
            breakdown['percentages'] = {
                'detection': detection_sum / total_loss * 100,
                'diffusion': breakdown['diffusion']['weighted'] / total_loss * 100,
            }
            if self.use_perceptual:
                breakdown['percentages']['perceptual'] = breakdown['perceptual']['raw'] / total_loss * 100

        return breakdown


if __name__ == "__main__":
    # Test the loss function
    print("Testing DiffV2X Loss...")

    # Dummy loss args
    args = {
        'num_class': 5,
        'pos_cls_weight': 2.0,
        'cls': {'weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        'reg': {'weight': 2.0},
        'dir': {'weight': 0.2, 'num_bins': 2},
        'depth': {'weight': 1.0},
        'diffusion': {
            'weight': 1.0,
            'use_perceptual': False,
            'perceptual_weight': 0.1,
        }
    }

    loss_func = DiffV2XLossMC(args)

    # Dummy output and target
    batch_size = 2
    H, W = 100, 352
    num_anchors = 2
    num_class = 5

    output_dict = {
        'cls_preds': torch.randn(batch_size, num_anchors * num_class * num_class, H, W),
        'reg_preds': torch.randn(batch_size, 7 * num_anchors * num_class, H, W),
        'dir_preds': torch.randn(batch_size, 2 * num_anchors * num_class, H, W),
        'diffusion_loss': torch.tensor(0.5),
    }

    target_dict = {
        'pos_equal_one': torch.randint(0, 2, (batch_size, H, W, num_anchors * num_class)),
        'neg_equal_one': torch.randint(0, 2, (batch_size, H, W, num_anchors * num_class)),
        'targets': torch.randn(batch_size, H, W, num_anchors * num_class, 7),
    }

    print("Loss function created successfully!")
    print(f"Breakdown: {loss_func.get_loss_breakdown()}")
