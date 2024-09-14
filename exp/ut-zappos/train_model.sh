#!/bin/bash

pwd_dir=$pwd
cd ../../

eval "$(conda shell.bash hook)"
conda activate clip

GPU_ID=$1
TAG=$2  # [csp, csp_noctx, proda_template]
MODEL=$3  # [csp, proda, coop]

# train CSP on UT-Zappos
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py \
  --config config/ut-zappos/${TAG}/train.yml \
  --experiment_name ${MODEL}

cd $pwd_dir
echo "Training finished!"