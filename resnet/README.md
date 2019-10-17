# Image Classification (ResNet)



### Prepare Data

```
sudo mount path-to-data-drive /mnt/

```

### Build Docker Image

```
docker build --pull -t mlperf-lambda:resnet .
```


### Run Benchmark

```

# LAMBDASYSTEM=Lambda_dual_basic ./run.sub
DATADIR=/mnt/imagenet CONT=mlperf-lambda:resnet PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

# Interactive mode
DATADIR=/mnt/imagenet; docker run -v $DATADIR:/imn --runtime=nvidia -t -i mlperf-lambda:resnet
```