# Translation (Transformer)

### Prepare Dataset

```

virtualenv -p /usr/bin/python3.6 venv-compile
. venv-compile/bin/activate
pip install torch==1.0.0 torchvision==0.2.2
python setup.py install
pip install mlperf_compliance==0.0.6

./run_preprocessing.sh

cp build/lib.linux-x86_64-3.6/fairseq/data/*.so fairseq/data/
cp build/lib.linux-x86_64-3.6/fairseq/data/*.py fairseq/data/

./run_conversion.sh
deactivate
rm -rf build
rm fairseq/data/*.so
```

### Build Docker Image

```
docker build --pull -t mlperf-lambda:transformer .
```

### Run Benchmark

```
DATADIR=/home/ubuntu/data/mlperf/transformer/examples/transformer/wmt14_en_de/utf8 PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
```
