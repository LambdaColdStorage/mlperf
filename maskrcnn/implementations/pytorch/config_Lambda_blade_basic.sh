#!/bin/bash

## DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.06"
               "SOLVER.MAX_ITER"      "80000"
               "SOLVER.WARMUP_FACTOR" "0.000096"
               "SOLVER.WARMUP_ITERS"  "625"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(24000, 32000)"
               "SOLVER.IMS_PER_BATCH"  "32"
               "TEST.IMS_PER_BATCH" "4"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "6000"
               "NHWC" "True"   
             )

## System run parms
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=07:00:00

## System config params
LAMBDANGPU=8
LAMBDASOCKETCORES=10
LAMBDANSOCKET=1
LAMBDAHT=2         # HT is on is 2, HT off is 1
LAMBDAIBDEVICES=''
BIND_LAUNCH=1
