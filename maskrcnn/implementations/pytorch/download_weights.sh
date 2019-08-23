#!/bin/bash

wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mkdir -p /home/ubuntu/data/mlperf/object_detection/coco2017/models
mv R-50.pkl /home/ubuntu/data/mlperf/object_detection/coco2017/models

