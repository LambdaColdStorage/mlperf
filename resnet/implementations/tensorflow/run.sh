#/bin/bash

LAMBDANGPU=${1:-"1"}
TRAIN_BATCH_SIZE=${2:-"64"}

RANDOM_SEED=$3
QUALITY=$4
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL_DIR="/tmp/resnet_imagenet_${RANDOM_SEED}"


python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn  \
  --model_dir $MODEL_DIR --train_epochs 1 --stop_threshold $QUALITY --batch_size $TRAIN_BATCH_SIZE \
  --version 1 --resnet_size 50 --epochs_between_evals 1 --num_gpus $LAMBDANGPU --dtype fp16
