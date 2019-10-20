__Lambda_dual_basic.sh__

```
ubuntu@lambda:~/mlperf/minigo/implementations/tensorflow$ NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
cat: /etc/LAMBDA-release: No such file or directory
                                                                                                                                                
================
== TensorFlow ==
================

NVIDIA Release 19.05 (build 6390160)
TensorFlow Version 1.13.1

Container image Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
Copyright 2017-2019 The TensorFlow Authors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.

NOTE: MOFED driver for multi-node communication was not detected.
      Multi-node communication performance may be reduced.

                                                                                                                                                
================
== TensorFlow ==
================

NVIDIA Release 19.05 (build 6390160)
TensorFlow Version 1.13.1

Container image Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
Copyright 2017-2019 The TensorFlow Authors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.

NOTE: MOFED driver for multi-node communication was not detected.
      Multi-node communication performance may be reduced.

Beginning trial 1 of 1
Gathering sys log on lambda
:::MLL 1571433686.379 submission_benchmark: {"metadata": {"lineno": 179, "file": "mlperf_log_utils.py"}, "value": "minigo"}
:::MLL 1571433686.379 submission_org: {"metadata": {"lineno": 184, "file": "mlperf_log_utils.py"}, "value": "Lambda Labs, Inc"}
WARNING: Log validation: Key "submission_division" is not in known minigo keys.
:::MLL 1571433686.380 submission_division: {"metadata": {"lineno": 188, "file": "mlperf_log_utils.py"}, "value": "closed"}
:::MLL 1571433686.381 submission_status: {"metadata": {"lineno": 192, "file": "mlperf_log_utils.py"}, "value": "onprem"}
:::MLL 1571433686.381 submission_platform: {"metadata": {"lineno": 196, "file": "mlperf_log_utils.py"}, "value": "1xTo Be Filled By O.E.M."}
:::MLL 1571433686.382 submission_entry: {"metadata": {"lineno": 200, "file": "mlperf_log_utils.py"}, "value": "{'notes': 'N/A', 'nodes': \"{'num_network_cards': '0', 'num_cores': '6', 'sys_storage_size': '1x 42.8M + 1x 465.8G + 1x 4.2M + 1x 89.1M + 2x 3.7M + 1x 89M + 1x 13M + 1x 149.9M + 1x 1.9T + 2x 54.5M + 1x 4M + 1x 44.2M + 2x 956K + 1x 140.9M + 1x 14.8M + 1x 140.7M', 'accelerator': 'Quadro RTX 8000', 'cpu': '1x Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz', 'network_card': '', 'notes': '', 'num_vcpus': '12', 'num_accelerators': '2', 'sys_mem_size': '31 GB', 'cpu_accel_interconnect': 'QPI', 'num_nodes': '1', 'sys_storage_type': '<unknown bus> SSD'}\", 'compilers': 'gcc (Ubuntu 5.4.0-6ubuntu1~16.04.11) 5.4.0 20160609', 'hardware': 'To Be Filled By O.E.M.', 'power': 'N/A', 'interconnect': ' ', 'os': 'Ubuntu 18.04.3 LTS / ', 'framework': 'TensorFlow NVIDIA Release 19.05', 'libraries': \"{'openmpi_version': '3.1.3', 'trt_version': '5.1.5.0', 'cuda_driver_version': '418.67', 'mofed_version': '0', 'container_base': 'Ubuntu-16.04', 'nccl_version': '2.4.6', 'cublas_version': '10.2.0.163', 'cuda_version': '10.1.163', 'dali_version': '0.9.1', 'cudnn_version': '7.6.0.64'}\"}"}
:::MLL 1571433686.382 submission_poc_name: {"metadata": {"lineno": 204, "file": "mlperf_log_utils.py"}, "value": "Chuan Li"}
:::MLL 1571433686.383 submission_poc_email: {"metadata": {"lineno": 208, "file": "mlperf_log_utils.py"}, "value": "c@lambdalabs.com"}
Clearing caches
:::MLL 1571433687.370 cache_clear: {"metadata": {"lineno": 1, "file": "<string>"}, "value": true}
Launching on node lambda
+ pids+=($!)
+ set +x
++ eval echo
+++ echo
+ docker exec -e LAMBDASYSTEM=Lambda_dual_basic -e 'SEED=     8000716' -e 'MULTI_NODE= --master_port=4847' -e SLURM_JOB_ID=191018142052126499923 -e SLURM_NTASKS_PER_NODE=2 -e SLURM_NNODES=1 -e 'MLPERF_HOST_OS=Ubuntu 18.04.3 LTS / ' cont_191018142052126499923 ./run_and_time.sh
Run vars: id 191018142052126499923 gpus 2 mparams  --master_port=4847
Making dir ml_perf/checkpoint
I1018 21:21:28.079041 139851450840832 utils.py:74] Running: gsutil  -m  cp  -r  gs://minigo-pub/ml_perf/checkpoint/9  ml_perf/checkpoint
I1018 21:22:24.472113 139851450840832 utils.py:84] gsutil finished: 56.393 seconds
Making dir ml_perf/target
I1018 21:22:24.473378 139851450840832 utils.py:74] Running: gsutil  -m  cp  -r  gs://minigo-pub/ml_perf/target/9  ml_perf/target
I1018 21:22:26.161779 139851450840832 utils.py:84] gsutil finished: 1.688 seconds
I1018 21:22:26.162992 139851450840832 utils.py:74] Running: python  freeze_graph.py  --model_path=ml_perf/target/9/target  --trt_batch=2048  --use_tpu=False
I1018 21:22:45.032048 139851450840832 utils.py:84] freeze_graph finished: 18.869 seconds
I1018 21:22:45.033927 139851450840832 utils.py:74] Running: python  freeze_graph.py  --model_path=ml_perf/checkpoint/9/work_dir/model.ckpt-9383  --trt_batch=2048  --use_tpu=False
I1018 21:23:01.382500 139851450840832 utils.py:84] freeze_graph finished: 16.348 seconds
STARTING TIMING RUN AT 2019-10-18 09:23:01 PM
running benchmark
+ echo 'running benchmark'
+ python ml_perf/reference_implementation.py --base_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23 --num_gpus_selfplay=2 --num_gpus_train=2 --num_socket=1 --cores_per_socket=6 --flagfile=ml_perf/flags/9/rl_loop.flags
Wiping dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/sgf/eval
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/work_dir
:::MLL 1571433783.159 init_start: {"metadata": {"lineno": 879, "file": "ml_perf/reference_implementation.py"}, "value": null}
100%|██████████| 1/1 [00:02<00:00,  2.36s/it]
100%|██████████| 1/1 [00:02<00:00,  2.25s/it]
100%|██████████| 1/1 [00:02<00:00,  2.05s/it]
100%|██████████| 1/1 [00:02<00:00,  2.29s/it]
100%|██████████| 1/1 [00:02<00:00,  2.28s/it]
100%|██████████| 1/1 [00:02<00:00,  2.08s/it]
100%|██████████| 1/1 [00:02<00:00,  2.03s/it]
100%|██████████| 1/1 [00:02<00:00,  2.32s/it]
100%|██████████| 1/1 [00:02<00:00,  2.05s/it]
100%|██████████| 1/1 [00:02<00:00,  2.35s/it]Got 389066 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000003.tfrecord.zz: 7.014 seconds
Got 380455 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000005.tfrecord.zz: 7.015 seconds
Got 347710 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000006.tfrecord.zz: 6.314 seconds
Got 391566 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000000.tfrecord.zz: 7.216 seconds
Got 377794 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000004.tfrecord.zz: 7.013 seconds
Got 348491 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000007.tfrecord.zz: 6.213 seconds
Got 343332 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000009.tfrecord.zz: 6.013 seconds
Got 383972 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000002.tfrecord.zz: 6.915 seconds
Got 345218 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000008.tfrecord.zz: 6.214 seconds
Got 388115 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000001.tfrecord.zz: 7.216 seconds
:::MLL 1571433878.253 opt_base_learning_rate: {"metadata": {"lineno": 901, "file": "ml_perf/reference_implementation.py"}, "value": [0.32, 0.032, 0.0032]}
:::MLL 1571433878.254 opt_learning_rate_decay_boundary_steps: {"metadata": {"lineno": 902, "file": "ml_perf/reference_implementation.py"}, "value": [12500, 18750]}
:::MLL 1571433878.254 global_batch_size: {"metadata": {"lineno": 903, "file": "ml_perf/reference_implementation.py"}, "value": 8192}
:::MLL 1571433878.254 virtual_losses: {"metadata": {"lineno": 906, "file": "ml_perf/reference_implementation.py"}, "value": 8}
:::MLL 1571433878.255 init_stop: {"metadata": {"lineno": 988, "file": "ml_perf/reference_implementation.py"}, "value": null}
:::MLL 1571433878.255 run_start: {"metadata": {"lineno": 989, "file": "ml_perf/reference_implementation.py"}, "value": null}
:::MLL 1571433878.256 train_loop: {"metadata": {"lineno": 992, "file": "ml_perf/reference_implementation.py"}, "value": null}
:::MLL 1571433878.256 epoch_start: {"metadata": {"lineno": 993, "epoch_num": 1, "file": "ml_perf/reference_implementation.py"}, "value": null}

I1018 21:24:38.257962 140541503809280 utils.py:74] Running: mpiexec  --allow-run-as-root  --map-by  ppr:2:socket,pe=2  -np  2  python3  train.py  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000009.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000009.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000008.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000008.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000007.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000007.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000006.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000006.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000005.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000005.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000004.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000004.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000003.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000003.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000002.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000002.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000001.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000001.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000000.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000000.tfrecord.zz_0  --conv_width=32  --fc_width=64  --trunk_layers=9  --value_cost_weight=0.25  --summary_steps=64  --shuffle_buffer_size=10000  --filter_amount=0.5  --train_batch_size=8192  --lr_rates=[0.32,0.032,0.0032]  --lr_boundaries=[12500,18750]  --l2_strength=0.0001  --work_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/work_dir  --export_path=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000001-000001  --training_seed=2  --use_mgpu_horovod=true  --freeze=true
I1018 21:24:38.277956 140541503809280 utils.py:74] Running: numactl  --physcpubind=4-4  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000001-000000  --seed=8
I1018 21:24:38.300566 140541503809280 utils.py:74] Running: numactl  --physcpubind=5-5  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000001-000000  --seed=9
I1018 21:24:38.326795 140541503809280 utils.py:74] Running: numactl  --physcpubind=10-10  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000001-000000  --seed=10
I1018 21:24:38.352412 140541503809280 utils.py:74] Running: numactl  --physcpubind=11-11  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000001-000000  --seed=11
I1018 21:25:28.038456 140541503809280 utils.py:84] numactl finished: 49.738 seconds
I1018 21:25:28.427662 140541503809280 utils.py:84] numactl finished: 50.101 seconds
I1018 21:25:28.921284 140541503809280 utils.py:84] numactl finished: 50.643 seconds
I1018 21:25:30.354240 140541503809280 utils.py:84] numactl finished: 52.001 seconds
I1018 21:25:30.355540 140541503809280 reference_implementation.py:364] Thread 10 stopping
Played 32 games, total time 50.0859 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      20      15       5       0      12       4       8       0
Ran 8957 batches with an average size of 81.638
I1018 21:25:30.356164 140541503809280 reference_implementation.py:364] Thread 1 stopping
Played 32 games, total time 49.0542 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      16      14       2       0      16       6      10       0
Ran 7985 batches with an average size of 91.202
I1018 21:25:30.356256 140541503809280 reference_implementation.py:364] Thread 0 stopping
Played 32 games, total time 49.5015 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      18      14       4       0      14      12       2       0
Ran 8117 batches with an average size of 93.1221
I1018 21:25:30.356346 140541503809280 reference_implementation.py:364] Thread 10 stopping
Played 32 games, total time 51.5221 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      10       6       4       0      22      16       6       0
Ran 8875 batches with an average size of 79.1094
I1018 21:25:30.356421 140541503809280 reference_implementation.py:372] Black won 0.500, white won 0.500
I1018 21:25:30.357014 140541503809280 reference_implementation.py:379] Writing golden chunk from "/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000001-000000/*/*.zz"
100%|██████████| 125/125 [00:00<00:00, 434.39it/s]
I1018 21:26:11.135101 140541503809280 utils.py:84] train finished: 92.876 seconds
Train get_golden_chunks at iter = 1 has win_size = 10
Got 11665 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000001-000000.tfrecord.zz: 0.302 seconds
:::MLL 1571433971.141 save_model: {"metadata": {"lineno": 519, "file": "ml_perf/reference_implementation.py"}, "value": {"iteration": 1}}
I1018 21:26:11.141598 140541503809280 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000001-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/sgf/eval/000001-000001  --seed=2
I1018 21:26:11.169556 140541503809280 utils.py:74] Running: python3  freeze_graph.py  --model_path=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000001-000001  --trt_batch=1024
I1018 21:26:28.168682 140541503809280 utils.py:84] freeze_graph finished: 16.999 seconds
I1018 21:26:53.479601 140541503809280 utils.py:84] w finished: 42.338 seconds
I1018 21:26:53.481138 140541503809280 reference_implementation.py:625] Evaluated 100 games, total time 39.93703715s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000001-000001.pb      11       6       5       0       7       2       5       0
checkpoint.pb         43      43       0       0      39      33       6       0
Ran 3684 batches with an average size of 314.375
Ran 3722 batches with an average size of 313.618
I1018 21:26:53.481280 140541503809280 reference_implementation.py:630] Win rate 000001-000001.pb vs checkpoint.pb: 0.180
:::MLL 1571434013.482 epoch_stop: {"metadata": {"lineno": 1010, "epoch_num": 1, "file": "ml_perf/reference_implementation.py"}, "value": null}
I1018 21:26:53.482251 140541503809280 utils.py:84] Iteration time: 135.228 seconds
:::MLL 1571434013.483 epoch_start: {"metadata": {"lineno": 993, "epoch_num": 2, "file": "ml_perf/reference_implementation.py"}, "value": null}
I1018 21:26:53.484742 140541503809280 utils.py:74] Running: mpiexec  --allow-run-as-root  --map-by  ppr:2:socket,pe=2  -np  2  python3  train.py  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000001-000000.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000001-000000.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000009.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000009.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000008.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000008.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000007.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000007.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000006.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000006.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000005.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000005.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000004.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000004.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000003.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000003.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000002.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000002.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000001.tfrecord.zz_1  /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000000-000001.tfrecord.zz_0  --conv_width=32  --fc_width=64  --trunk_layers=9  --value_cost_weight=0.25  --summary_steps=64  --shuffle_buffer_size=10000  --filter_amount=0.5  --train_batch_size=8192  --lr_rates=[0.32,0.032,0.0032]  --lr_boundaries=[12500,18750]  --l2_strength=0.0001  --work_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/work_dir  --export_path=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000002-000001  --training_seed=3  --use_mgpu_horovod=true  --freeze=true
I1018 21:26:53.512987 140541503809280 utils.py:74] Running: numactl  --physcpubind=4-4  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000002-000000  --seed=12
I1018 21:26:53.540275 140541503809280 utils.py:74] Running: numactl  --physcpubind=5-5  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000002-000000  --seed=13
I1018 21:26:53.577108 140541503809280 utils.py:74] Running: numactl  --physcpubind=10-10  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000002-000000  --seed=14
I1018 21:26:53.617181 140541503809280 utils.py:74] Running: numactl  --physcpubind=11-11  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/holdout/000002-000000  --seed=15
I1018 21:27:43.988392 140541503809280 utils.py:84] numactl finished: 50.411 seconds
I1018 21:27:44.450734 140541503809280 utils.py:84] numactl finished: 50.937 seconds
I1018 21:27:45.079691 140541503809280 utils.py:84] numactl finished: 51.462 seconds
I1018 21:27:47.275565 140541503809280 utils.py:84] numactl finished: 53.735 seconds
I1018 21:27:47.278459 140541503809280 reference_implementation.py:364] Thread 1 stopping
Played 32 games, total time 50.547 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      16      14       2       0      16       4      12       0
Ran 8966 batches with an average size of 81.11
I1018 21:27:47.278749 140541503809280 reference_implementation.py:364] Thread 6 stopping
Played 32 games, total time 53.3949 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      18       6      12       0      14       7       7       0
Ran 9187 batches with an average size of 77.4477
I1018 21:27:47.278949 140541503809280 reference_implementation.py:364] Thread 11 stopping
Played 32 games, total time 49.937 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      20      17       3       0      12       6       6       0
Ran 8367 batches with an average size of 86.4577
I1018 21:27:47.279130 140541503809280 reference_implementation.py:364] Thread 1 stopping
Played 32 games, total time 51.04 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      26      16      10       0       6       4       2       0
Ran 8112 batches with an average size of 90.6075
I1018 21:27:47.279305 140541503809280 reference_implementation.py:372] Black won 0.625, white won 0.375
I1018 21:27:47.279646 140541503809280 reference_implementation.py:379] Writing golden chunk from "/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/selfplay/000002-000000/*/*.zz"
100%|██████████| 122/122 [00:00<00:00, 314.41it/s]
I1018 21:28:25.794373 140541503809280 utils.py:84] train finished: 92.309 seconds
Train get_golden_chunks at iter = 2 has win_size = 10
Got 11385 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/data/golden_chunks/000002-000000.tfrecord.zz: 0.301 seconds
:::MLL 1571434105.796 save_model: {"metadata": {"lineno": 519, "file": "ml_perf/reference_implementation.py"}, "value": {"iteration": 2}}
I1018 21:28:25.796803 140541503809280 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000002-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/checkpoint.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/sgf/eval/000002-000001  --seed=3
I1018 21:28:25.826036 140541503809280 utils.py:74] Running: python3  freeze_graph.py  --model_path=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000002-000001  --trt_batch=1024
I1018 21:28:42.594294 140541503809280 utils.py:84] freeze_graph finished: 16.768 seconds
I1018 21:29:17.397016 140541503809280 utils.py:84] w finished: 51.600 seconds
I1018 21:29:17.399079 140541503809280 reference_implementation.py:625] Evaluated 100 games, total time 49.147739756s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000002-000001.pb      20      15       5       0      19       8      11       0
checkpoint.pb         31      11      20       0      30       1      29       0
Ran 4737 batches with an average size of 276.681
Ran 4746 batches with an average size of 275.4
I1018 21:29:17.399371 140541503809280 reference_implementation.py:630] Win rate 000002-000001.pb vs checkpoint.pb: 0.390
:::MLL 1571434157.401 epoch_stop: {"metadata": {"lineno": 1010, "epoch_num": 2, "file": "ml_perf/reference_implementation.py"}, "value": null}
I1018 21:29:17.401213 140541503809280 utils.py:84] Iteration time: 143.919 seconds
I1018 21:29:17.401580 140541503809280 utils.py:84] Total time from mpi_rank=0: 374.371 seconds
++ date +%s
+ end=1571434157
++ date '+%Y-%m-%d %r'
+ end_fmt='2019-10-18 09:29:17 PM'
+ echo 'ENDING TIMING RUN AT 2019-10-18 09:29:17 PM'
ENDING TIMING RUN AT 2019-10-18 09:29:17 PM
+ python ml_perf/eval_models.py --base_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23 --num_gpus_selfplay=2 --num_gpus_train=2 --num_socket=1 --cores_per_socket=6 --flags_dir=ml_perf/flags/9/
:::MLL 1571434159.295 eval_start: {"metadata": {"lineno": 55, "file": "ml_perf/eval_models.py", "epoch_num": 1}, "value": null}
I1018 21:29:19.295943 140238302668544 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000001-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/target.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/sgf/eval/target  --seed=1
I1018 21:29:54.917487 140238302668544 utils.py:84] w finished: 35.621 seconds
I1018 21:29:54.918728 140238302668544 reference_implementation.py:625] Evaluated 100 games, total time 34.181387556s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000001-000001.pb       1       1       0       0       0       0       0       0
target.pb             50      50       0       0      49      42       7       0
Ran 3115 batches with an average size of 358.779
Ran 3149 batches with an average size of 353.682
I1018 21:29:54.919290 140238302668544 reference_implementation.py:630] Win rate 000001-000001.pb vs target.pb: 0.010
:::MLL 1571434194.920 eval_accuracy: {"metadata": {"lineno": 60, "file": "ml_perf/eval_models.py", "epoch_num": 1}, "value": 0.01}
:::MLL 1571434194.920 eval_stop: {"metadata": {"lineno": 61, "file": "ml_perf/eval_models.py", "epoch_num": 1}, "value": null}
:::MLL 1571434194.921 eval_start: {"metadata": {"lineno": 55, "file": "ml_perf/eval_models.py", "epoch_num": 2}, "value": null}
I1018 21:29:54.921201 140238302668544 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/000002-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/models/target.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-18-21-23/sgf/eval/target  --seed=2
I1018 21:30:34.734204 140238302668544 utils.py:84] w finished: 39.813 seconds
I1018 21:30:34.736686 140238302668544 reference_implementation.py:625] Evaluated 100 games, total time 37.511920308s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000002-000001.pb       2       1       1       0       2       0       2       0
target.pb             48      27      21       0      48       0      48       0
Ran 3449 batches with an average size of 331.612
Ran 3457 batches with an average size of 331.442
I1018 21:30:34.737031 140238302668544 reference_implementation.py:630] Win rate 000002-000001.pb vs target.pb: 0.040
:::MLL 1571434234.739 eval_accuracy: {"metadata": {"lineno": 60, "file": "ml_perf/eval_models.py", "epoch_num": 2}, "value": 0.04}
:::MLL 1571434234.740 eval_stop: {"metadata": {"lineno": 61, "file": "ml_perf/eval_models.py", "epoch_num": 2}, "value": null}
:::MLL 1571434234.741 eval_result: {"metadata": {"timestamp": 0, "lineno": 68, "iteration": 2, "file": "ml_perf/eval_models.py"}, "value": null}
+ ret_code=0
+ set +x
RESULT,REINFORCEMENT,376,nvidia,2019-10-18 09:23:01 PM
cont_191018142052126499923
```