# Object Detection (MaskRCNN)


### Prepare Data

The data is the same as the ssd benchmark.

Need to download pre-trained backbone model.

```
./download_weights.sh
```


### Build Docker Image

```
docker build --pull -t mlperf-lambda:object_detection .
```

### Run Benchmark

```
DATADIR=/home/ubuntu/data/mlperf/object_detection PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
```
