<div align="center">

# LR-V2X

### Loss-Resilient Collaborative Perception under Low-Bandwidth Communication

**NeurIPS 2026**

Kang Yang<sup>1</sup> · Tianci Bu<sup>2</sup> · Peng Wang<sup>1</sup> · Deying Li<sup>1</sup> · Yongcai Wang<sup>1,3,*</sup>

<sup>1</sup>Renmin University of China &nbsp; <sup>2</sup>HKUST<br>
<sup>3</sup>Hebei Key Laboratory of Real-virtual Integrated Autonomous Systems (RIAS)<br>
<sup>*</sup>Corresponding author: [ycw@ruc.edu.cn](mailto:ycw@ruc.edu.cn)

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-2b7a78)](#overview)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-ee4c2c)](#installation)
[![Citation](https://img.shields.io/badge/Citation-BibTeX-546e7a)](#citation)

**[Overview](#overview) · [Results](#results) · [Installation](#installation) · [Training](#training) · [Evaluation](#evaluation)**

Paper link: coming soon. Pretrained checkpoints will be released separately.

<img src="assets/teaser.png" alt="LR-V2X reconstructs collaborator BEV features from compact messages under packet loss" width="900">

</div>

## Overview

LR-V2X reconstructs collaborator BEV features **before fusion**, so collaborative
LiDAR detection can work with compact messages even when most packets are lost.
Each collaborator transmits a **128 KiB latent** instead of an 8 MiB dense BEV map.
The receiver combines the surviving latent with its own BEV context and relative
geometry to recover the collaborator feature.

1. **Compress:** a latent encoder maps the collaborator BEV to a compact message.
2. **Reconstruct:** a latent prior decoder (LPD) provides an initial BEV map.
   An ego-conditioned DiT reconstructs it with a **single denoising forward pass**.
3. **Fuse and detect:** pyramid fusion combines the recovered collaborator feature
   with the ego feature for 3D detection.

Training uses complete communication. Packet erasure is introduced at evaluation;
the ego feature remains available locally.

<p align="center">
  <img src="assets/method.png" alt="Latent compression, ego-conditioned reconstruction, and pyramid fusion" width="1000">
</p>

### Included Code

A compact LiDAR-only implementation for **DAIR-V2X** and **V2XREAL**:

- LR-V2X models, three training stages, and packet-loss evaluation.
- Attn, F-Cooper, CoBEVT, V2X-ViT, HEAL, and the codebook-based CodeFilling baseline.
- Dataset loaders, PointPillars, fusion modules, losses, and detection post-processing.

Datasets, checkpoints, experiment logs, profiling tools, and analysis scripts
are not bundled.

## Results

Overall mAP (%) at **90% packet loss**, from Table 1 of the paper.
Payload is measured per collaborator frame **before loss**.

### DAIR-V2X

| Method | Payload | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|:--|--:|--:|--:|--:|
| Attn | 8 MiB | 66.87 | 66.20 | 52.56 |
| F-Cooper | 8 MiB | 67.89 | 65.90 | 52.87 |
| CoBEVT | 8 MiB | 68.83 | 63.55 | 44.47 |
| V2X-ViT | 8 MiB | 68.78 | 64.33 | 50.97 |
| HEAL | 8 MiB | 64.43 | 61.68 | 50.50 |
| CodeFilling | 128 KiB | 63.38 | 60.93 | 50.62 |
| **LR-V2X** | **128 KiB** | **69.35** | **66.90** | **53.37** |

### V2XREAL

| Method | Payload | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|:--|--:|--:|--:|--:|
| Attn | 8 MiB | 42.55 | 35.54 | 20.33 |
| F-Cooper | 8 MiB | 42.49 | 34.30 | 17.21 |
| CoBEVT | 8 MiB | 38.92 | 30.93 | 15.64 |
| V2X-ViT | 8 MiB | 40.73 | 32.29 | 18.93 |
| HEAL | 8 MiB | 43.07 | 37.64 | 21.79 |
| CodeFilling | 128 KiB | 40.54 | 36.69 | 23.16 |
| **LR-V2X** | **128 KiB** | **44.87** | **38.60** | **24.18** |

The latent contains `16 × 32 × 64` float32 values: 131,072 bytes (128 KiB),
a **64× smaller payload** than the dense BEV. The paper labels these binary
tensor sizes as KB/MB; this README uses KiB/MiB explicitly.

<p align="center">
  <img src="assets/sweep.png" alt="Detection accuracy across packet-loss rates on DAIR-V2X and V2XREAL" width="1000">
</p>

<p align="center">
  <img src="assets/qual.png" alt="Qualitative collaborative detection comparisons from the paper appendix" width="1000">
</p>

## Installation

Run commands from the repository root. The reference environment uses Linux,
Python 3.8, PyTorch 1.12.0, torchvision 0.13.0, and spconv 2.3.6.

```bash
git clone https://github.com/sidiangongyuan/LR-V2X.git
cd LR-V2X

conda create -n lrv2x python=3.8 -y
conda activate lrv2x
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
pip install spconv-cu117==2.3.6
pip install -e . --no-deps
python opencood/utils/setup.py build_ext --inplace
```

Use a CUDA-capable GPU and an NVIDIA driver compatible with the CUDA runtimes.
The Cython extension requires a C/C++ compiler. The released PointPillars
configurations do not require the optional OpenPCDet CUDA extensions.

## Datasets

Obtain the datasets from their official providers:

- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X): vehicle-infrastructure LiDAR pairs.
- [V2X-Real](https://github.com/ucla-mobility/V2X-Real): the LiDAR-only V2V setting,
  with vehicle, pedestrian, and truck detection.

Use the OpenCOOD/HEAL-preprocessed layout and the benchmark split files:

```text
data/
├── dairv2x/
│   ├── train.json
│   ├── val.json
│   ├── cooperative/
│   ├── infrastructure-side/
│   └── vehicle-side/
└── v2xreal/
    ├── train/
    ├── val/
    └── test/
```

Edit `data_dir`, `root_dir`, `validate_dir`, and `test_dir` in the YAML
files if your data is elsewhere. DAIR-V2X uses its validation split for
evaluation. V2XREAL training validates on `val/` and evaluates on `test/`.
The V2XREAL modality assignment is included in
`opencood/modality_assign/v2xreal_4modality.json`.

## Training

The LR-V2X configurations follow Appendix A:

| Stage | Trainable components | Epochs | Batch size | Learning rate |
|:--|:--|--:|--:|--:|
| 1 | PointPillars, pyramid fusion, detection heads | 40 | 8 | 0.002 |
| 2 | Latent encoder, LPD, DiT | 100 | 12 | 0.0002 |
| 3 | Latent encoder, LPD, DiT, fusion, heads | 20 | 8 | 0.0002 |

Stage 3 keeps the sensor encoder/backbone frozen and uses
`detection loss + 0.1 × reconstruction loss`. Inference uses one DiT step
at the fixed noise timestep `700`.

```bash
DATASET=dairv2x  # or v2xreal

python -m opencood.tools.diffv2x_stages.train_diffv2x_stage1 \
  --hypes_yaml configs/$DATASET/stage1.yaml --model_dir logs/$DATASET/stage1

STAGE1=$(find logs/$DATASET/stage1 -name 'net_epoch_bestval_at*.pth' -print -quit)
python -m opencood.tools.diffv2x_stages.train_diffv2x_stage2 \
  --hypes_yaml configs/$DATASET/stage2.yaml --stage1_model "$STAGE1" \
  --model_dir logs/$DATASET/stage2

STAGE2=$(find logs/$DATASET/stage2 -name 'net_epoch_bestval_at*.pth' -print -quit)
python -m opencood.tools.diffv2x_stages.train_diffv2x_stage3 \
  --hypes_yaml configs/$DATASET/stage3.yaml --stage2_model "$STAGE2" \
  --model_dir logs/$DATASET/stage3
```

Each stage keeps its lowest-validation-loss checkpoint. Use a fresh output
directory for each stage. YAML batch sizes are **per process**; when using
`torchrun`, divide them by the GPU count to keep the same global batch size.

### Baselines

The five feature-fusion baselines use the common training entrypoint:

```bash
python -m opencood.tools.train \
  --hypes_yaml configs/dairv2x/baselines/cobevt.yaml --model_dir logs/dairv2x/cobevt
```

Replace `cobevt` with `attn`, `fcooper`, `v2xvit`, or `heal`, and
use `configs/v2xreal/` for V2XREAL.

CodeFilling uses its three codebook stages:

```bash
DATASET=dairv2x
python -m opencood.tools.train \
  --hypes_yaml configs/$DATASET/baselines/codefilling/stage1.yaml \
  --model_dir logs/$DATASET/codefilling1

STAGE1=$(find logs/$DATASET/codefilling1 -name 'net_epoch_bestval_at*.pth' -print -quit)
python -m opencood.tools.train_stage2 \
  --hypes_yaml configs/$DATASET/baselines/codefilling/stage2.yaml \
  --stage1_model "$STAGE1" --model_dir logs/$DATASET/codefilling2

STAGE2=$(find logs/$DATASET/codefilling2 -name 'net_epoch_bestval_at*.pth' -print -quit)
python -m opencood.tools.train_stage3 \
  --hypes_yaml configs/$DATASET/baselines/codefilling/stage3.yaml \
  --stage2_model "$STAGE2" --model_dir logs/$DATASET/codefilling3
```

## Evaluation

Use the same evaluator for LR-V2X and all included baselines:

```bash
# 90% packet loss
python -m opencood.tools.evaluate \
  --model_dir logs/dairv2x/stage3 --retention 0.1

# Complete communication and the paper's packet-loss sweep
python -m opencood.tools.evaluate \
  --model_dir logs/v2xreal/stage3 --retention 1.0 0.9 0.7 0.5 0.3 0.1

# Spatial-burst loss
python -m opencood.tools.evaluate \
  --model_dir logs/dairv2x/stage3 --retention 0.1 --packet_loss_mode burst
```

`--retention` is the fraction of spatial packets that arrive, **not**
a compression setting. It does not change the transmitted latent dimensions.
The evaluator loads `config.yaml` and the best-validation checkpoint from
`--model_dir`. It writes range-wise AP for DAIR-V2X or class-wise AP and
mAP for V2XREAL to `evaluation.csv`. AP values in the CSV are fractions;
the tables above report percentages. FPS covers model forward and detection
post-processing, excluding data loading and metric accumulation.

## Citation

```bibtex
@inproceedings{yang2026lrv2x,
  title     = {LR-V2X: Loss-Resilient Collaborative Perception under Low-Bandwidth Communication},
  author    = {Yang, Kang and Bu, Tianci and Wang, Peng and Li, Deying and Wang, Yongcai},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements and License

The implementation builds on
[OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD),
[HEAL](https://github.com/yifanlu0227/HEAL), and
[DiT](https://github.com/facebookresearch/DiT). Please also credit the methods
and datasets used in your experiments.

Original LR-V2X contributions identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
use [MIT](LICENSE-LR-V2X). Upstream and derived files retain their original
terms, including the [Academic Software License](LICENSE) and DiT's
[CC BY-NC 4.0](licenses/DiT-CC-BY-NC-4.0.txt). This is **not** a blanket MIT
license for the whole repository.
