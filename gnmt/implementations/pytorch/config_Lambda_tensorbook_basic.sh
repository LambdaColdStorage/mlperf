#!/bin/bash

## System run parms
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=5:00:00

## DL params
LR=${LR:-"0.25e-3"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
WARMUP_STEPS=${WARMUP_STEPS:-200}
REMAIN_STEPS=${REMAIN_STEPS:-6453}
DECAY_INTERVAL=${DECAY_INTERVAL:-809}
TARGET=${TARGET:-24.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
NUMEPOCHS=${NUMEPOCHS:-8}
MATH=${MATH:-amp_fp16}
EXTRA_OPTS=${EXTRA_OPTS-"\
   --fused-attention \
   --fused-xentropy \
   --no-log-all-ranks \
   "}

## System config params
LAMBDANGPU=1
LAMBDASOCKETCORES=10
LAMBDAHT=2 	# HT is on is 2, HT off is 1
LAMBDAIBDEVICES=''
LAMBDANSOCKET=1
BIND_LAUNCH=1
