import unittest
from unittest import mock

import opencood.data_utils.datasets as datasets


class DatasetFactoryCalibrateTest(unittest.TestCase):
    @staticmethod
    def _dataset_cfg():
        return {
            'fusion': {
                'core_method': 'intermediateheter',
                'dataset': 'dairv2x',
            },
        }

    def test_omits_calibrate_for_legacy_fusion_dataset(self):
        class LegacyDataset:
            def __init__(self, params, visualize, train=True):
                self.train = train

        with mock.patch.object(
            datasets,
            'getIntermediateheterFusionDataset',
            return_value=LegacyDataset,
        ):
            result = datasets.build_dataset(
                self._dataset_cfg(),
                visualize=False,
                train=False,
                calibrate=True,
            )

        self.assertFalse(result.train)

    def test_passes_calibrate_when_supported(self):
        class CalibratedDataset:
            def __init__(self, params, visualize, train=True, calibrate=False):
                self.calibrate = calibrate

        with mock.patch.object(
            datasets,
            'getIntermediateheterFusionDataset',
            return_value=CalibratedDataset,
        ):
            result = datasets.build_dataset(
                self._dataset_cfg(),
                visualize=False,
                train=False,
                calibrate=True,
            )

        self.assertTrue(result.calibrate)


if __name__ == '__main__':
    unittest.main()
