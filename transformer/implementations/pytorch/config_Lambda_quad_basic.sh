#!/bin/bash

## DL params
MAX_TOKENS=10240
LEARNING_RATE="1.0e-3"
WARMUP_UPDATES=1000
EXTRA_PARAMS="--max-source-positions 64 --max-target-positions 64 --enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 105404416 --parallel-backward-allred-cuda-nstreams 2 --adam-betas (0.9,0.98) "

## System run parms
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=02:00:00

## System config params
LAMBDANGPU=4
LAMBDASOCKETCORES=10
LAMBDANSOCKET=1
LAMBDAHT=2         # HT is on is 2, HT off is 1
LAMBDAIBDEVICES=''
