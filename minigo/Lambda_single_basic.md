__Lambda_single_basic.sh__

```
ubuntu@lambda:~/mlperf/minigo/implementations/tensorflow$ NEXP=1 CONT=mlperf-lambda:minigo PULL=0 LAMBDASYSTEM=Lambda_single_basic ./run.sub
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
:::MLL 1571444841.158 submission_benchmark: {"value": "minigo", "metadata": {"lineno": 179, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.159 submission_org: {"value": "Lambda Labs, Inc", "metadata": {"lineno": 184, "file": "mlperf_log_utils.py"}}
WARNING: Log validation: Key "submission_division" is not in known minigo keys.
:::MLL 1571444841.159 submission_division: {"value": "closed", "metadata": {"lineno": 188, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.160 submission_status: {"value": "onprem", "metadata": {"lineno": 192, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.161 submission_platform: {"value": "1xTo Be Filled By O.E.M.", "metadata": {"lineno": 196, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.162 submission_entry: {"value": "{'framework': 'TensorFlow NVIDIA Release 19.05', 'os': 'Ubuntu 18.04.3 LTS / ', 'notes': 'N/A', 'interconnect': ' ', 'libraries': \"{'dali_version': '0.9.1', 'container_base': 'Ubuntu-16.04', 'cuda_version': '10.1.163', 'openmpi_version': '3.1.3', 'nccl_version': '2.4.6', 'cudnn_version': '7.6.0.64', 'mofed_version': '0', 'cublas_version': '10.2.0.163', 'cuda_driver_version': '418.67', 'trt_version': '5.1.5.0'}\", 'hardware': 'To Be Filled By O.E.M.', 'compilers': 'gcc (Ubuntu 5.4.0-6ubuntu1~16.04.11) 5.4.0 20160609', 'nodes': \"{'cpu': '1x Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz', 'accelerator': 'Quadro RTX 8000', 'notes': '', 'num_nodes': '1', 'sys_storage_size': '2x 956K + 1x 42.8M + 2x 54.5M + 1x 4M + 1x 89M + 1x 4.2M + 2x 3.7M + 1x 140.7M + 1x 140.9M + 1x 13M + 1x 44.2M + 1x 1.9T + 1x 149.9M + 1x 465.8G + 1x 14.8M + 1x 89.1M', 'cpu_accel_interconnect': 'QPI', 'num_cores': '6', 'num_vcpus': '12', 'num_accelerators': '1', 'sys_mem_size': '31 GB', 'network_card': '', 'num_network_cards': '0', 'sys_storage_type': '<unknown bus> SSD'}\", 'power': 'N/A'}", "metadata": {"lineno": 200, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.162 submission_poc_name: {"value": "Chuan Li", "metadata": {"lineno": 204, "file": "mlperf_log_utils.py"}}
:::MLL 1571444841.163 submission_poc_email: {"value": "c@lambdalabs.com", "metadata": {"lineno": 208, "file": "mlperf_log_utils.py"}}
Clearing caches
:::MLL 1571444842.568 cache_clear: {"metadata": {"file": "<string>", "lineno": 1}, "value": true}
Launching on node lambda
+ pids+=($!)
+ set +x
++ eval echo
+++ echo
+ docker exec -e LAMBDASYSTEM=Lambda_single_basic -e 'SEED=    14539163' -e 'MULTI_NODE= --master_port=4398' -e SLURM_JOB_ID=191018172647232968532 -e SLURM_NTASKS_PER_NODE=1 -e SLURM_NNODES=1 -e 'MLPERF_HOST_OS=Ubuntu 18.04.3 LTS / ' cont_191018172647232968532 ./run_and_time.sh
Run vars: id 191018172647232968532 gpus 1 mparams  --master_port=4398
Making dir ml_perf/checkpoint
I1019 00:27:23.303431 140690541410048 utils.py:74] Running: gsutil  -m  cp  -r  gs://minigo-pub/ml_perf/checkpoint/9  ml_perf/checkpoint
I1019 00:28:19.280176 140690541410048 utils.py:84] gsutil finished: 55.976 seconds
Making dir ml_perf/target
I1019 00:28:19.281675 140690541410048 utils.py:74] Running: gsutil  -m  cp  -r  gs://minigo-pub/ml_perf/target/9  ml_perf/target
I1019 00:28:21.193974 140690541410048 utils.py:84] gsutil finished: 1.912 seconds
I1019 00:28:21.194544 140690541410048 utils.py:74] Running: python  freeze_graph.py  --model_path=ml_perf/target/9/target  --trt_batch=2048  --use_tpu=False
I1019 00:28:39.684361 140690541410048 utils.py:84] freeze_graph finished: 18.490 seconds
I1019 00:28:39.685254 140690541410048 utils.py:74] Running: python  freeze_graph.py  --model_path=ml_perf/checkpoint/9/work_dir/model.ckpt-9383  --trt_batch=2048  --use_tpu=False
I1019 00:28:55.300148 140690541410048 utils.py:84] freeze_graph finished: 15.615 seconds
STARTING TIMING RUN AT 2019-10-19 12:28:55 AM
running benchmark
+ echo 'running benchmark'
+ python ml_perf/reference_implementation.py --base_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28 --num_gpus_selfplay=1 --num_gpus_train=1 --num_socket=1 --cores_per_socket=6 --flagfile=ml_perf/flags/9/rl_loop.flags
Wiping dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/holdout
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/sgf/eval
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks
Making dir /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/work_dir
:::MLL 1571444936.960 init_start: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 879}}
100%|██████████| 1/1 [00:02<00:00,  2.35s/it]
100%|██████████| 1/1 [00:02<00:00,  2.28s/it]
100%|██████████| 1/1 [00:01<00:00,  1.95s/it]
100%|██████████| 1/1 [00:02<00:00,  2.37s/it]
100%|██████████| 1/1 [00:02<00:00,  2.29s/it]
100%|██████████| 1/1 [00:02<00:00,  2.08s/it]
100%|██████████| 1/1 [00:02<00:00,  2.03s/it]
100%|██████████| 1/1 [00:02<00:00,  2.29s/it]
100%|██████████| 1/1 [00:02<00:00,  2.10s/it]
100%|██████████| 1/1 [00:02<00:00,  2.31s/it]Got 387700 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000003.tfrecord.zz: 12.726 seconds
Got 380569 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000005.tfrecord.zz: 12.424 seconds
Got 347447 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000006.tfrecord.zz: 11.122 seconds
Got 391504 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000000.tfrecord.zz: 12.928 seconds
Got 377794 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000004.tfrecord.zz: 12.426 seconds
Got 348718 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000007.tfrecord.zz: 11.222 seconds
Got 343567 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000009.tfrecord.zz: 10.922 seconds
Got 383972 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000002.tfrecord.zz: 12.626 seconds
Got 346336 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000008.tfrecord.zz: 11.122 seconds
Got 388250 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000001.tfrecord.zz: 12.825 seconds
:::MLL 1571445085.383 opt_base_learning_rate: {"value": [0.32, 0.032, 0.0032], "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 901}}
:::MLL 1571445085.384 opt_learning_rate_decay_boundary_steps: {"value": [12500, 18750], "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 902}}
:::MLL 1571445085.384 global_batch_size: {"value": 8192, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 903}}
:::MLL 1571445085.384 virtual_losses: {"value": 8, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 906}}
:::MLL 1571445085.385 init_stop: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 988}}
:::MLL 1571445085.385 run_start: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 989}}
:::MLL 1571445085.386 train_loop: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 992}}
:::MLL 1571445085.386 epoch_start: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "epoch_num": 1, "lineno": 993}}

I1019 00:31:25.388040 140563099174656 utils.py:74] Running: mpiexec  --allow-run-as-root  --map-by  ppr:1:socket,pe=2  -np  1  python3  train.py  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000009.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000008.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000007.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000006.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000005.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000004.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000003.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000002.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000001.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000000.tfrecord.zz_0  --conv_width=32  --fc_width=64  --trunk_layers=9  --value_cost_weight=0.25  --summary_steps=64  --shuffle_buffer_size=10000  --filter_amount=0.5  --train_batch_size=8192  --lr_rates=[0.32,0.032,0.0032]  --lr_boundaries=[12500,18750]  --l2_strength=0.0001  --work_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/work_dir  --export_path=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000001-000001  --training_seed=2  --use_mgpu_horovod=true  --freeze=true
I1019 00:31:25.417310 140563099174656 utils.py:74] Running: numactl  --physcpubind=2-5  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/holdout/000001-000000  --seed=4
I1019 00:31:25.440349 140563099174656 utils.py:74] Running: numactl  --physcpubind=8-11  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000001-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/holdout/000001-000000  --seed=5
I1019 00:31:58.587513 140563099174656 utils.py:84] numactl finished: 33.147 seconds
I1019 00:32:02.248715 140563099174656 utils.py:84] numactl finished: 36.831 seconds
I1019 00:32:02.250058 140563099174656 reference_implementation.py:364] Thread 1 stopping
Played 32 games, total time 36.2425 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      24      17       7       0       8       2       6       0
Ran 7172 batches with an average size of 98.5086
I1019 00:32:02.250696 140563099174656 reference_implementation.py:364] Thread 4 stopping
Played 32 games, total time 32.54 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      15      11       4       0      17      11       6       0
Ran 6333 batches with an average size of 99.7947
I1019 00:32:02.250790 140563099174656 reference_implementation.py:372] Black won 0.609, white won 0.391
I1019 00:32:02.251379 140563099174656 reference_implementation.py:379] Writing golden chunk from "/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000001-000000/*/*.zz"
100%|██████████| 62/62 [00:00<00:00, 428.39it/s]
I1019 00:33:37.837938 140563099174656 utils.py:84] train finished: 132.449 seconds
Train get_golden_chunks at iter = 1 has win_size = 10
Got 5420 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000001-000000.tfrecord.zz: 0.201 seconds
:::MLL 1571445217.844 save_model: {"value": {"iteration": 1}, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 519}}
I1019 00:33:37.844590 140563099174656 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000001-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/sgf/eval/000001-000001  --seed=2
I1019 00:33:37.869017 140563099174656 utils.py:74] Running: python3  freeze_graph.py  --model_path=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000001-000001  --trt_batch=1024
I1019 00:33:54.826295 140563099174656 utils.py:84] freeze_graph finished: 16.957 seconds
I1019 00:34:11.546458 140563099174656 utils.py:84] w finished: 33.702 seconds
I1019 00:34:11.548709 140563099174656 reference_implementation.py:625] Evaluated 100 games, total time 32.229688422s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000001-000001.pb       3       2       1       0       0       0       0       0
checkpoint.pb         50      50       0       0      47      47       0       0
Ran 3112 batches with an average size of 283.769
Ran 2755 batches with an average size of 296.608
I1019 00:34:11.549093 140563099174656 reference_implementation.py:630] Win rate 000001-000001.pb vs checkpoint.pb: 0.030
:::MLL 1571445251.551 epoch_stop: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "epoch_num": 1, "lineno": 1010}}
I1019 00:34:11.551219 140563099174656 utils.py:84] Iteration time: 166.167 seconds
:::MLL 1571445251.552 epoch_start: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "epoch_num": 2, "lineno": 993}}
I1019 00:34:11.555444 140563099174656 utils.py:74] Running: mpiexec  --allow-run-as-root  --map-by  ppr:1:socket,pe=2  -np  1  python3  train.py  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000001-000000.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000009.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000008.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000007.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000006.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000005.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000004.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000003.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000002.tfrecord.zz_0  /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000000-000001.tfrecord.zz_0  --conv_width=32  --fc_width=64  --trunk_layers=9  --value_cost_weight=0.25  --summary_steps=64  --shuffle_buffer_size=10000  --filter_amount=0.5  --train_batch_size=8192  --lr_rates=[0.32,0.032,0.0032]  --lr_boundaries=[12500,18750]  --l2_strength=0.0001  --work_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/work_dir  --export_path=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000002-000001  --training_seed=3  --use_mgpu_horovod=true  --freeze=true
I1019 00:34:11.590601 140563099174656 utils.py:74] Running: numactl  --physcpubind=2-5  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/holdout/000002-000000  --seed=6
I1019 00:34:11.628282 140563099174656 utils.py:74] Running: numactl  --physcpubind=8-11  --membind=0  bazel-bin/cc/selfplay  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --model=trt:1024,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb  --output_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000002-000000  --holdout_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/holdout/000002-000000  --seed=7
I1019 00:34:45.153395 140563099174656 utils.py:84] numactl finished: 33.562 seconds
I1019 00:34:50.819060 140563099174656 utils.py:84] numactl finished: 39.190 seconds
I1019 00:34:50.822245 140563099174656 reference_implementation.py:364] Thread 15 stopping
Played 32 games, total time 33.1484 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      24      17       7       0       8       6       2       0
Ran 6266 batches with an average size of 113.775
I1019 00:34:50.822609 140563099174656 reference_implementation.py:364] Thread 13 stopping
Played 32 games, total time 38.8244 sec.
             Black   Black   Black   Black   White   White   White   White
             total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
checkpoint      20      15       5       0      12       7       5       0
Ran 7804 batches with an average size of 91.1799
I1019 00:34:50.822843 140563099174656 reference_implementation.py:372] Black won 0.688, white won 0.312
I1019 00:34:50.823246 140563099174656 reference_implementation.py:379] Writing golden chunk from "/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/selfplay/000002-000000/*/*.zz"
100%|██████████| 60/60 [00:00<00:00, 465.80it/s]
I1019 00:36:11.219604 140563099174656 utils.py:84] train finished: 119.664 seconds
Train get_golden_chunks at iter = 2 has win_size = 10
Got 5507 examples
Writing examples to /opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/data/golden_chunks/000002-000000.tfrecord.zz: 0.304 seconds
:::MLL 1571445371.221 save_model: {"value": {"iteration": 2}, "metadata": {"file": "ml_perf/reference_implementation.py", "lineno": 519}}
I1019 00:36:11.221797 140563099174656 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000002-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/checkpoint.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/sgf/eval/000002-000001  --seed=3
I1019 00:36:11.252013 140563099174656 utils.py:74] Running: python3  freeze_graph.py  --model_path=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000002-000001  --trt_batch=1024
I1019 00:36:17.538780 140563099174656 utils.py:84] w finished: 6.317 seconds
I1019 00:36:17.540806 140563099174656 reference_implementation.py:625] Evaluated 100 games, total time 4.918628174s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000002-000001.pb       0       0       0       0       0       0       0       0
checkpoint.pb         50       0      50       0      50      50       0       0
Ran 56 batches with an average size of 165.161
Ran 154 batches with an average size of 183.747
I1019 00:36:17.541062 140563099174656 reference_implementation.py:630] Win rate 000002-000001.pb vs checkpoint.pb: 0.000
I1019 00:36:26.478516 140563099174656 utils.py:84] freeze_graph finished: 15.226 seconds
:::MLL 1571445386.481 epoch_stop: {"value": null, "metadata": {"file": "ml_perf/reference_implementation.py", "epoch_num": 2, "lineno": 1010}}
I1019 00:36:26.481391 140563099174656 utils.py:84] Iteration time: 134.930 seconds
I1019 00:36:26.481925 140563099174656 utils.py:84] Total time from mpi_rank=0: 449.641 seconds
++ date +%s
+ end=1571445386
++ date '+%Y-%m-%d %r'
+ end_fmt='2019-10-19 12:36:26 AM'
+ echo 'ENDING TIMING RUN AT 2019-10-19 12:36:26 AM'
ENDING TIMING RUN AT 2019-10-19 12:36:26 AM
+ python ml_perf/eval_models.py --base_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28 --num_gpus_selfplay=1 --num_gpus_train=1 --num_socket=1 --cores_per_socket=6 --flags_dir=ml_perf/flags/9/
:::MLL 1571445388.461 eval_start: {"metadata": {"lineno": 55, "epoch_num": 1, "file": "ml_perf/eval_models.py"}, "value": null}
I1019 00:36:28.462376 139624527431424 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000001-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/target.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/sgf/eval/target  --seed=1
I1019 00:37:06.305882 139624527431424 utils.py:84] w finished: 37.843 seconds
I1019 00:37:06.307366 139624527431424 reference_implementation.py:625] Evaluated 100 games, total time 36.443903453s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000001-000001.pb       0       0       0       0       0       0       0       0
target.pb             50      45       5       0      50      50       0       0
Ran 3032 batches with an average size of 318.578
Ran 3807 batches with an average size of 292.37
I1019 00:37:06.308104 139624527431424 reference_implementation.py:630] Win rate 000001-000001.pb vs target.pb: 0.000
:::MLL 1571445426.309 eval_accuracy: {"metadata": {"lineno": 60, "epoch_num": 1, "file": "ml_perf/eval_models.py"}, "value": 0.0}
:::MLL 1571445426.310 eval_stop: {"metadata": {"lineno": 61, "epoch_num": 1, "file": "ml_perf/eval_models.py"}, "value": null}
:::MLL 1571445426.310 eval_start: {"metadata": {"lineno": 55, "epoch_num": 2, "file": "ml_perf/eval_models.py"}, "value": null}
I1019 00:37:06.310620 139624527431424 utils.py:74] Running: ./w.sh  numactl  --physcpubind=0-5,6-11  --membind=0  bazel-bin/cc/eval  --num_games=32  --num_readouts=240  --value_init_penalty=0.2  --holdout_pct=0.03  --disable_resign_pct=0.1  --resign_threshold=-0.99  --parallel_games=16  --virtual_losses=8  --num_games=100  --parallel_games=100  --model=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/000002-000001.pb.og  --model_two=tf,/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/models/target.pb.og  --sgf_dir=/opt/reinforcement/minigo/results/lambda-2019-10-19-00-28/sgf/eval/target  --seed=2
I1019 00:37:21.065974 139624527431424 utils.py:84] w finished: 14.755 seconds
I1019 00:37:21.068459 139624527431424 reference_implementation.py:625] Evaluated 100 games, total time 12.527477709s
                   Black   Black   Black   Black   White   White   White   White
                   total   passes  resign  m.lmt.  total   passes  resign  m.lmt.
000002-000001.pb       0       0       0       0       0       0       0       0
target.pb             50       0      50       0      50      50       0       0
Ran 1195 batches with an average size of 167.932
Ran 683 batches with an average size of 160.161
I1019 00:37:21.068811 139624527431424 reference_implementation.py:630] Win rate 000002-000001.pb vs target.pb: 0.000
:::MLL 1571445441.070 eval_accuracy: {"metadata": {"lineno": 60, "epoch_num": 2, "file": "ml_perf/eval_models.py"}, "value": 0.0}
:::MLL 1571445441.071 eval_stop: {"metadata": {"lineno": 61, "epoch_num": 2, "file": "ml_perf/eval_models.py"}, "value": null}
:::MLL 1571445441.072 eval_result: {"metadata": {"file": "ml_perf/eval_models.py", "timestamp": 0, "iteration": 2, "lineno": 68}, "value": null}
+ ret_code=0
+ set +x
RESULT,REINFORCEMENT,451,nvidia,2019-10-19 12:28:55 AM
cont_191018172647232968532
```