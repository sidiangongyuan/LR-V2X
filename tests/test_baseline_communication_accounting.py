import unittest

import torch

from opencood.models.heter_model_baseline import _dense_communication_stats


class DenseCommunicationAccountingTest(unittest.TestCase):
    def test_payload_is_per_transmitted_collaborator_before_loss(self):
        feature = torch.zeros((3, 4, 2, 3), dtype=torch.float32)

        stats = _dense_communication_stats(feature, torch.tensor([1, 2]))

        self.assertEqual(stats['num_collaborators'], 1)
        self.assertEqual(stats['comm_payload_bytes'], 96.0)
        self.assertEqual(stats['scene_payload_bytes'], [0, 96])
        self.assertEqual(stats['comm_rate'], 1.0)

    def test_scene_payload_scales_with_collaborator_count(self):
        feature = torch.zeros((5, 2, 2, 2), dtype=torch.float16)

        stats = _dense_communication_stats(feature, torch.tensor([2, 3]))

        self.assertEqual(stats['num_collaborators'], 3)
        self.assertEqual(stats['comm_payload_bytes'], 16.0)
        self.assertEqual(stats['scene_payload_bytes'], [16, 32])

    def test_no_collaborator_reports_zero_payload(self):
        feature = torch.zeros((2, 4, 2, 3), dtype=torch.float32)

        stats = _dense_communication_stats(feature, torch.tensor([1, 1]))

        self.assertEqual(stats['num_collaborators'], 0)
        self.assertEqual(stats['comm_payload_bytes'], 0.0)
        self.assertEqual(stats['comm_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
