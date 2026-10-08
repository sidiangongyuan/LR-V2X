import argparse
import unittest
from unittest import mock

import torch

from opencood.tools.evaluate import configure_packet_loss, update_class_stats
from opencood.loss.diffv2x_pyramid_loss import DiffV2XPyramidLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


class EvaluateTest(unittest.TestCase):
    def test_overrides_preserve_source_and_set_nested_retention(self):
        original = {'model': {'args': {'diffusion': {}, 'codebook': {}}}}
        opt = argparse.Namespace(packet_loss_mode='burst', burst_coarse_h=8,
                                 burst_coarse_w=16, mask_seed=42)
        result = configure_packet_loss(original, 0.1, opt)
        self.assertEqual(original['model']['args']['diffusion'], {})
        for key in ('diffusion', 'codebook'):
            self.assertEqual(result['model']['args'][key]['compression_ratio'], 0.1)

    def test_class_stats_count_ground_truth_when_no_predictions(self):
        stats = {'vehicle': {0.3: {'tp': [], 'fp': [], 'gt': 0}}}
        with mock.patch('opencood.tools.evaluate.eval_utils_mc.caluclate_tp_fp') as update:
            update_class_stats(stats, None, None, torch.zeros(2, 8, 3), torch.tensor([1, 2]))
        self.assertEqual(update.call_count, 3)
        self.assertEqual(update.call_args.args[2].shape[0], 1)

    def test_single_class_loss_adds_weighted_reconstruction(self):
        criterion = DiffV2XPyramidLoss.__new__(DiffV2XPyramidLoss)
        torch.nn.Module.__init__(criterion)
        criterion.diffusion_weight = 0.1
        criterion.loss_dict = {}
        reconstruction = torch.tensor(3.0, requires_grad=True)
        with mock.patch.object(PointPillarPyramidLoss, 'forward', return_value=torch.tensor(2.0)):
            loss = criterion({'diffusion_loss': reconstruction}, {})
        self.assertAlmostEqual(loss.item(), 2.3, places=5)
        loss.backward()
        self.assertAlmostEqual(reconstruction.grad.item(), 0.1, places=5)


if __name__ == '__main__':
    unittest.main()
