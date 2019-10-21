#!/bin/bash

# System run params
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# DL params
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}

# System config params
LAMBDANGPU=8
LAMBDASOCKETCORES=20
LAMBDANSOCKET=2
LAMBDAHT=2
LAMBDAIBDEVICES=''

