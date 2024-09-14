#!/bin/bash

pwd_dir=$pwd
cd ../../

eval "$(conda shell.bash hook)"
conda activate clip

GPU_ID=$1

# train Model on MIT-States
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py \
  --config config/mit-states/fullmodel/train.yml \
  --experiment_name gencsp

cd $pwd_dir
echo "Training finished!"