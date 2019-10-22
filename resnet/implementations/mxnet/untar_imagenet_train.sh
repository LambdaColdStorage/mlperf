#!/bin/bash

for filename in /home/ubuntu/data/mlperf/imagenet-mxnet/train-tar/*.tar; do
	outputname="/home/ubuntu/data/mlperf/imagenet-mxnet/train-jpeg/$(basename "$filename" .tar)"
	mkdir -p $outputname
	tar -vxf $filename -C $outputname
done
