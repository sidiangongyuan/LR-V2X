import unittest

import torch

from opencood.utils.packet_loss_utils import (
    build_spatial_packet_loss_mask,
    contiguous_numeric_sequence_ends,
    encode_packet_loss_sample_index,
    encode_temporal_sample_index,
    grouped_sequence_ends,
)


class PacketLossMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def build(self, **overrides):
        kwargs = {
            "record_len": [2],
            "spatial_size": (32, 64),
            "keep_ratio": 0.1,
            "device": self.device,
            "dtype": self.dtype,
            "mode": "bernoulli",
            "sample_indices": [7],
            "seed_base": 42,
        }
        kwargs.update(overrides)
        return build_spatial_packet_loss_mask(**kwargs)

    def test_ego_is_never_dropped(self) -> None:
        mask = self.build(keep_ratio=0.0)
        self.assertTrue(torch.equal(mask[0], torch.ones_like(mask[0])))
        self.assertTrue(torch.equal(mask[1], torch.zeros_like(mask[1])))

    def test_seeded_bernoulli_is_reproducible(self) -> None:
        first = self.build()
        second = self.build()
        self.assertTrue(torch.equal(first, second))

    def test_seeded_bernoulli_changes_across_samples(self) -> None:
        first = self.build(sample_indices=[7])
        second = self.build(sample_indices=[8])
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_non_temporal_index_tracks_dataset_index(self) -> None:
        self.assertEqual(
            encode_packet_loss_sample_index(
                dataset_index=17,
                mode="bernoulli",
                sequence_end_indices=[20],
                temporal_block_len=5,
            ),
            17,
        )

    def test_temporal_mask_is_constant_within_block(self) -> None:
        first = self.build(
            mode="temporal_block",
            temporal_block_len=5,
            sample_indices=[10],
        )
        last = self.build(
            mode="temporal_block",
            temporal_block_len=5,
            sample_indices=[14],
        )
        self.assertTrue(torch.equal(first, last))

    def test_temporal_mask_changes_across_blocks(self) -> None:
        first = self.build(
            mode="temporal_block",
            temporal_block_len=5,
            sample_indices=[10],
        )
        next_block = self.build(
            mode="temporal_block",
            temporal_block_len=5,
            sample_indices=[15],
        )
        self.assertFalse(torch.equal(first[1], next_block[1]))

    def test_temporal_mode_requires_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires seed_base"):
            self.build(
                mode="temporal_block",
                temporal_block_len=5,
                seed_base=None,
            )

    def test_temporal_keep_ratio_matches_expectation(self) -> None:
        kept = []
        for sample_idx in range(200):
            mask = self.build(
                spatial_size=(16, 32),
                mode="temporal_block",
                temporal_block_len=5,
                sample_indices=[sample_idx],
            )
            kept.append(float(mask[1].mean()))
        self.assertAlmostEqual(sum(kept) / len(kept), 0.1, delta=0.015)

    def test_temporal_index_resets_at_sequence_boundary(self) -> None:
        sequence_ends = [6, 12]
        encoded_last = encode_temporal_sample_index(5, sequence_ends, 5)
        encoded_next = encode_temporal_sample_index(6, sequence_ends, 5)

        self.assertNotEqual(encoded_last // 5, encoded_next // 5)
        self.assertEqual(
            encode_temporal_sample_index(6, sequence_ends, 5),
            encode_temporal_sample_index(10, sequence_ends, 5),
        )

    def test_contiguous_numeric_sequence_ends(self) -> None:
        frame_ids = ["000211", "000212", "000214", "000215", "000216", "000871"]
        self.assertEqual(
            contiguous_numeric_sequence_ends(frame_ids),
            [2, 5, 6],
        )

    def test_grouped_sequence_ends_uses_sequence_labels(self) -> None:
        self.assertEqual(
            grouped_sequence_ends(["55", "55", "55", "45", "45", "106"]),
            [3, 5, 6],
        )

    def test_burst_mode_remains_spatially_coarse(self) -> None:
        mask = self.build(
            mode="burst",
            burst_coarse_shape=(4, 8),
        )[1, 0]
        for row in range(4):
            for col in range(8):
                block = mask[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
                self.assertEqual(int(torch.unique(block).numel()), 1)


if __name__ == "__main__":
    unittest.main()
