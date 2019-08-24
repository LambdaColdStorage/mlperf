#!/bin/bash

# System run params
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# DL params
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}

# System config params
LAMBDANGPU=8
LAMBDASOCKETCORES=10
LAMBDANSOCKET=1
LAMBDAHT=2
LAMBDAIBDEVICES=''

