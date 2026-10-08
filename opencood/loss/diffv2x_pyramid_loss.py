"""Detection and reconstruction loss for the single-class DAIR-V2X model."""

from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


class DiffV2XPyramidLoss(PointPillarPyramidLoss):
    def __init__(self, args: dict):
        super().__init__(args)
        self.diffusion_weight = args.get('diffusion', {}).get('weight', 0.0)

    def forward(self, output_dict: dict, target_dict: dict, suffix: str = ''):
        detection_loss = super().forward(output_dict, target_dict, suffix)
        diffusion_loss = output_dict.get('diffusion_loss', detection_loss.new_zeros(()))
        total_loss = detection_loss + self.diffusion_weight * diffusion_loss
        self.loss_dict.update({
            'diffusion_loss': float(diffusion_loss.detach()),
            'total_loss': float(total_loss.detach()),
        })
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None, suffix=''):
        super().logging(epoch, batch_id, batch_len, writer, pbar, suffix)
        if writer is not None:
            writer.add_scalar('Diffusion_loss' + suffix, self.loss_dict['diffusion_loss'],
                              epoch * batch_len + batch_id)
