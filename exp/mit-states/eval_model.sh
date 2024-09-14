#!/bin/bash

pwd_dir=$pwd
cd ../../

eval "$(conda shell.bash hook)"
conda activate clip

GPU_ID=$1
TAG=$2 # closed / open

case ${TAG} in
  closed)
  # closed-set evaluation
  echo "=> Closed World Evaluation:"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u evaluate.py \
    --config config/mit-states/fullmodel/eval.yml \
    --experiment_name gencsp
  ;;
  open)
  # open-world evaluation
  echo "=> Open World Evaluation:"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u evaluate.py \
    --config config/mit-states/fullmodel/eval.yml \
    --experiment_name gencsp \
    --open_world
  ;;
esac

cd $pwd_dir
echo "Evaluation finished!"