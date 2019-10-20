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
NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_tensorbook_basic ./run.sub

NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub

NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub

NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_quad_basic ./run.sub

NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_blade_basic ./run.sub

NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_hyperplane_basic ./run.sub
```

### Important Settings

__config.sh__
- LAMBDASOCKETCORES: Need to be set as total_num_gpu_cores / num_gpus

- __NEXP__ number of trails for running the benchmark. Set to 1 to save time.

__ml_perf/flags/9/bootstrap.flags__
- num_games: "Total number of games to play. Only one of num_games and run_forever must be set. "
- parallel_games: "Number of games to play in parallel." Has to be smaller than num_games. Default to be half of the the num_games.

__ml_perf/flags/9/rl_loop.flags__
- iterations: "Number of iterations of the RL loop.""

__ml_perf/flags/9/train.flags__
- train_batch_size: training batch size. Will be splitted across GPUs. 

__Total number of Games__

```
num_games * num_gpu * 2, so the number of games natually scale up with number of GPUs.
```

### Key Metric

__Iteration time__: The total time for training + evaluation for a single epoch.




