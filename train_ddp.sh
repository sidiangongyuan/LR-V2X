#!/bin/bash

LOGFILE="train_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env \
    opencood/tools/train_ddp.py \
    --hypes_yaml /mnt/sdb/public/data/yk/projects/QuantV2X/opencood/hypes_yaml/v2x_real/DiffV2X/diffv2x_lidar.yaml \
    2>&1 | tee $LOGFILE
