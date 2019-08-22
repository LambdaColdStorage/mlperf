# Reinforcement Learning Benchmark (Minigo)

### Prepare data
No data needs to be prepared.

### Build Docker Image

```
cd implementations/tensorflow/minigo
docker build --pull -t mlperf-lambda:minigo .
```


### Run Benchmark

```
CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub

CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
```