# LR-V2X: Loss-Resilient Collaborative Perception under Low-Bandwidth Communication

> Official implementation of "LR-V2X: Loss-Resilient Collaborative Perception under Low-Bandwidth Communication" (NeurIPS 2026).

## Overview

LR-V2X addresses the practical challenge of collaborative perception under both **low-bandwidth** constraints and **stochastic packet loss**. When spatial packets are dropped, the receiver observes large holes in the transmitted feature maps. LR-V2X recovers missing BEV content from packet-corrupted latents via two stages:

1. **Latent Prior Decoder (LPD)**: Overlapping transposed convolutions extrapolate received locations into a spatially complete prior BEV map.
2. **Noise-Conditioned Reconstructor (NCR)**: A DiT-based denoiser refines the prior, conditioned on the received latent and the ego agent's own BEV context, following a diffusion-style training strategy that generalizes to unseen packet-loss rates at test time.

LR-V2X achieves the strongest overall mAP on both DAIR-V2X and V2XREAL under 90% packet loss while transmitting only 128 KB per collaborator (64× less than dense-BEV baselines).

## Installation

### Step 1: Basic Installation

```bash
conda create -n lrv2x python=3.8 pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate lrv2x
pip install -r requirements.txt
python setup.py develop
```

### Step 2: Install Spconv 2.x

Check the [spconv table](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library) for your CUDA version, e.g.:

```bash
pip install spconv-cu116
```

### Step 3: Compile IoU extension

```bash
python opencood/utils/setup.py build_ext --inplace
```

## Data Preparation

**V2XREAL**: Download from the [V2X-Real website](https://mobility-lab.seas.ucla.edu/v2x-real/). Organize as:
```
v2xreal/
├── train/
├── validate/
└── test/
```

**DAIR-V2X-C**: Download from [DAIR-V2X page](https://thudair.baai.ac.cn/index). Use complemented annotations following [this page](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).

Update the `root_dir`, `validate_dir`, and `test_dir` fields in the YAML config files under `opencood/hypes_yaml/`.

## Training

LR-V2X uses a three-stage training procedure:

### Stage 1 — Backbone pre-training
```bash
python opencood/tools/diffv2x_stages/train_diffv2x_stage1.py \
    --hypes_yaml opencood/hypes_yaml/v2x_real/DiffV2X_Stages/v2x_real_stage1_pyramid_Baseline.yaml \
    --model_dir logs/lrv2x_stage1
```

### Stage 2 — LPD + NCR training (diffusion module, backbone frozen)
```bash
python opencood/tools/diffv2x_stages/train_diffv2x_stage2.py \
    --hypes_yaml opencood/hypes_yaml/v2x_real/DiffV2X_Stages/v2x_real_diffv2x_stage2_latent_prior_8x.yaml \
    --stage1_model logs/lrv2x_stage1/net_epoch_bestval_at<N>.pth \
    --model_dir logs/lrv2x_stage2
```

### Stage 3 — End-to-end fine-tuning
```bash
python opencood/tools/diffv2x_stages/train_diffv2x_stage3.py \
    --hypes_yaml opencood/hypes_yaml/v2x_real/DiffV2X_Stages/v2x_real_diffv2x_stage3_latent_prior_finetune_8x.yaml \
    --stage2_model logs/lrv2x_stage2/net_epoch_bestval_at<N>.pth \
    --model_dir logs/lrv2x_stage3
```

Multi-GPU training (replace `--nproc_per_node` with your GPU count):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env \
    opencood/tools/diffv2x_stages/train_diffv2x_stage2.py \
    --hypes_yaml opencood/hypes_yaml/v2x_real/DiffV2X_Stages/v2x_real_diffv2x_stage2_latent_prior_8x.yaml \
    --stage1_model logs/lrv2x_stage1/net_epoch_bestval_at<N>.pth \
    --model_dir logs/lrv2x_stage2
```

Configs for 4× and 16× compression are also provided under `opencood/hypes_yaml/v2x_real/DiffV2X_Stages/` and `opencood/hypes_yaml/dairv2x/DiffV2X_Stages/`.

## Evaluation

Evaluate under packet loss (sweep from 0% to 90%):
```bash
python opencood/tools/inference_pkloss_mc.py \
    --model_dir logs/lrv2x_stage3
```

## Key Source Files

| File | Description |
|---|---|
| `opencood/models/diffv2x_pyramid_mc.py` | Main LR-V2X model |
| `opencood/models/sub_modules/simple_prior_decoder.py` | Latent Prior Decoder (LPD) |
| `opencood/models/sub_modules/diffusion_model_dit.py` | DiT-based Noise-Conditioned Reconstructor (NCR) |
| `opencood/models/sub_modules/latent_encoder.py` | Latent encoder |
| `opencood/models/sub_modules/diffusion_sampler.py` | Diffusion forward/reverse schedule |
| `opencood/utils/packet_loss_utils.py` | Spatial packet-loss mask generation |

## Acknowledgement

This codebase builds upon [HEAL](https://github.com/yifanlu0227/HEAL) and [V2X-Real](https://github.com/ucla-mobility/V2X-Real).
