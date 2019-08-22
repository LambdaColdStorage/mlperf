#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LAMBDASYSTEM=${LAMBDASYSTEM:-"LAMBDA1"}
if [[ -f config_${LAMBDASYSTEM}.sh ]]; then
  source config_${LAMBDASYSTEM}.sh
else
  source config_LAMBDA1.sh
  echo "Unknown system, assuming LAMBDA1"
fi
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$LAMBDANGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
NUMEPOCHS=${NUMEPOCHS:-80}

echo "running benchmark"

export DATASET_DIR="/data/coco2017"
export TORCH_MODEL_ZOO="/data/torchvision"

# run training
python -m bind_launch --nsockets_per_node ${LAMBDANSOCKET} \
                      --ncores_per_socket ${LAMBDASOCKETCORES} \
                      --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE \
 train.py \
  --use-fp16 \
  --nhwc \
  --pad-input \
  --jit \
  --delay-allreduce \
  --opt-loss \
  --epochs "${NUMEPOCHS}" \
  --warmup-factor 0 \
  --no-save \
  --threshold=0.23 \
  --data ${DATASET_DIR} \
  --evaluation 120000 160000 180000 200000 220000 240000 260000 280000 \
  ${EXTRA_PARAMS[@]} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
