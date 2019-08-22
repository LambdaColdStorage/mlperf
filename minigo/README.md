# Reinforcement Learning Benchmark (Minigo)

### Prepare data
No data needs to be prepared.

### Build Docker Image

```
cd implementations/tensorflow
docker build --pull -t mlperf-lambda:minigo .
```


### Run Benchmark

```
CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_tensorbook_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_quad_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_blade_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_hyperplane_basic ./run.sub
```
