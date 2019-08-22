#!/bin/bash

# DL params

# System run params
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )


# System config params
LAMBDANGPU=8
LAMBDASOCKETCORES=12
LAMBDANSOCKET=2
LAMBDAHT=2
LAMBDAIBDEVICES=''
