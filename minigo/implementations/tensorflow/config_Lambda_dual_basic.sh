#!/bin/bash

# DL params

# System run params
LAMBDANNODES=1
LAMBDASYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )


# System config params
LAMBDANGPU=2
LAMBDASOCKETCORES=6
LAMBDANSOCKET=1
LAMBDAHT=2
LAMBDAIBDEVICES=''
