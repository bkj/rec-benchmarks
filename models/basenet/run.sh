#!/bin/bash

# run.sh

CUDA_VISIBLE_DEVICES=0 python basenet-rec.py --use-cache

    --emb-dim 800 \
    --eval-interval 1 \
    --epochs 10 \
    --batch-size 256 | tee results/tmp5