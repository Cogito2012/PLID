#!/bin/bash

pwd_dir=$pwd
cd ../../

eval "$(conda shell.bash hook)"
conda activate clip

GPU_ID=$1
TAG=$2 # closed / open
EXPTAG=$3
MODEL=$4  # [csp, csp_noctx, proda, coop]

THRESH=0.49937106273612186  # csp on mit-states

case ${TAG} in
  closed)
  # closed-set evaluation
  echo "=> Closed World Evaluation:"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u evaluate.py \
    --config config/cgqa/${EXPTAG}/eval.yml \
    --experiment_name ${MODEL}
  ;;
  open)
  # open-world evaluation
  echo "=> Open World Evaluation:"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -u evaluate.py \
    --config config/cgqa/${EXPTAG}/eval.yml \
    --experiment_name ${MODEL} \
    --open_world
    # --threshold ${THRESH} \
  ;;
esac

cd $pwd_dir
echo "Evaluation finished!"