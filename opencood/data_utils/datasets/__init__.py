import inspect

from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_3class_dataset import getIntermediateheter3classFusionDataset
from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from opencood.data_utils.datasets.basedataset.v2xreal_basedataset import V2XREALBaseDataset

# the final range for evaluation
# GT_RANGE = [-102.4, -102.4, -6, 102.4, 102.4, 3]
# GT_RANGE = [-202.4, -202.4, -7, 202.4, 202.4, 2]
GT_RANGE = [-100, -40, -15, 100, 40, 15]
# GT_RANGE = [-70, -70, -7, 70, 70, 4]
# GT_RANGE = [-51.2, -51.2, -5, 51.2, 51.2, 2]
# The communication range for cavs
COM_RANGE = 50


def build_dataset(dataset_cfg: dict, visualize: bool = False, train: bool = True,
                  calibrate: bool = False):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']

    factories = {
        'intermediateheter': getIntermediateheterFusionDataset,
        'intermediateheter3class': getIntermediateheter3classFusionDataset,
    }
    bases = {'dairv2x': DAIRV2XBaseDataset, 'v2xreal': V2XREALBaseDataset}
    fusion_dataset_func = factories[fusion_name]
    base_dataset_cls = bases[dataset_name]

    dataset_cls = fusion_dataset_func(base_dataset_cls)
    dataset_kwargs = {
        'params': dataset_cfg,
        'visualize': visualize,
        'train': train,
    }
    if 'calibrate' in inspect.signature(dataset_cls.__init__).parameters:
        dataset_kwargs['calibrate'] = calibrate

    dataset = dataset_cls(**dataset_kwargs)

    return dataset
