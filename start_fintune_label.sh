#!/bin/bash
#SBATCH -o log.knee_mae_condition1_50_re_%A

## Pre-training MAE
condition=1
JOB_DIR='finetune/patch_4_condition_'${condition}_re
echo ${JOB_DIR}
data_path='data/MRNet-v1.0'
finetune='output/128_huge_knee_75/checkpoint-720.pth'
python main_finetune_test_label.py \
    --device cuda \
    --condition ${condition} \
    --output_dir ${JOB_DIR} \
    --data_path ${data_path} \
    --finetune ${finetune} \
    --batch_size 1 \
    --epochs 40 \
    --warmup_epochs 0 \
    --model vit_huge_patch4 \
