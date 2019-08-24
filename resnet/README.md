# Image Classification (ResNet)



### Prepare Data

Copy TFRecords to '/home/ubuntu/data/mlperf'


### Build Docker Image

```
docker build --pull -t mlperf-lambda:resnet .
```


### Run Benchmark

```

LAMBDASYSTEM=Lambda_dual_basic ./run.sub

# LAMBDASYSTEM=Lambda_dual_basic docker run -v $DATADIR:/imn --runtime=nvidia -t -i mlperf-lambda:resnet "./run_and_time.sh" $SEED | tee "$LOGDIR.log"

CONT=mlperf-lambda:resnet PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

```
