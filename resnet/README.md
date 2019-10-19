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
NEXP=1 DATADIR=/mnt/imagenet CONT=mlperf-lambda:resnet PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub
NEXP=1 DATADIR=/mnt/imagenet CONT=mlperf-lambda:resnet PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

# Interactive mode
DATADIR=/mnt/imagenet; docker run -v $DATADIR:/imn --runtime=nvidia -t -i mlperf-lambda:resnet
```


### Note
- __NEXP__ number of trails for running the benchmark. Set to 1 to save time.
- __TRAIN_BATCH_SIZE__ in the config file will be splitted by number of GPUs (See `per_device_batch_size` in `resnet_run_loop.py`)
- __--train_epochs__ in `run.sh` is set to `1` to save time.
- __--epochs_between_evals__ in `run.sh` is set to `1` so it works with `train_epochs=1`
- __NUM_IMAGES, NUM_TRAIN_FILES and NUM_VAL_FILES__ are set to small numbers.   
- __--target_acc--__ in `run_and_time.sh` is set to `0.0001` as we are not testing model accuracy.

### Performance