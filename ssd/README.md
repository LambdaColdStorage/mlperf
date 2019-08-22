# Object Detection (SSD)

### Prepare data

```
cd implementations/pytorch
./download_dataset.sh
```

### Build Docker Image

```
docker build --pull -t mlperf-lambda:ssd .
```


### Run Benchmark

```
DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_tensorbook_basic ./run.sub

DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub

DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_quad_basic ./run.sub

DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_blade_basic ./run.sub

DATADIR=/home/ubuntu/data/mlperf/object_detection CONT=mlperf-lambda:ssd PULL=0 LAMBDASYSTEM=Lambda_hyperplane_basic ./run.sub
```
