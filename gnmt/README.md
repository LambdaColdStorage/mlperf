# Translation (Google Neural Machine Translation)


### Prepare Data

```
./download_dataset.sh /home/ubuntu/data/mlperf/gnmt
```

### Build Docker Image

```
docker build --pull -t mlperf-lambda:gnmt . 
```


### Run Benchmark

```
DATADIR=/home/ubuntu/data/mlperf/gnmt PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
```
