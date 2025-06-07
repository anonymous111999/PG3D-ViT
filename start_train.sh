#!/bin/bash
#SBATCH -o log.knee_mae_%A

## Pre-training MAE
JOB_DIR='output/256_huge_knee'
IMAGENET_DIR='data/data_2d/'
label_csv='data/train_labels.csv'
python main_pretrain_test.py \
    --output_dir ${JOB_DIR} \
    --label_csv ${label_csv} \
    --data_path ${IMAGENET_DIR} \
    --batch_size 64 \
    --epochs 961 \
    --warmup_epochs 0 \
    --start_epoch 0 \
    --mask_ratio 0.75 \
    --model mae_vit_huge_patch4\
