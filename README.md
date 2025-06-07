# PG3D-ViT
# Anonymous Medical Image Classification Framework

This repository contains code for a two-stage medical image classification framework based on masked autoencoding and cross-attention prompt guidance.

## 🔧 Overview of Training Pipeline

### Step 1: MAE Pretraining

Run the following script to perform masked autoencoder (MAE) pretraining on 2D medical image slices:

```bash
bash start_train.sh

This step trains a Vision Transformer encoder using a self-supervised MAE strategy to learn general-purpose representations from unlabeled medical image data.

Step 2: Cross-Attention Fine-tuning
After pretraining, run the following script to perform supervised classification using cross-attention and lesion-context prompts:

```bash
bash start_fintune_label.sh

The following publicly available datasets are used in this project:

1. MRNet
Knee MRI dataset for abnormality, ACL tear, and meniscus tear classification.

Provided by Stanford Machine Learning Group.

Download link: https://stanfordmlgroup.github.io/competitions/mrnet/

2. MedMNIST v1 / v2
A lightweight benchmark for medical image classification in 2D and 3D.

Covers multiple modalities: X-ray, MRI, CT, ultrasound, etc.

Official website: https://medmnist.com

Code and data hosted at: https://github.com/MedMNIST/MedMNIST

Version 2 reference:

Yang J et al., MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023.
