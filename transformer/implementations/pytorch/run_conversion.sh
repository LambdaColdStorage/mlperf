#!/bin/bash

set -e

SEED=$1

python3 convert_utf8_to_fairseq_binary.py --data_dir /home/ubuntu/data/mlperf/transformer/examples/transformer/wmt14_en_de
