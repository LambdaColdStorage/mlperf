# Translation (Transformer)

### Prepare Dataset

```
virtualenv -p /usr/bin/python3.6 venv-compile
. venv-compile/bin/activate
pip install torch==1.2.0 torchvision==0.4.0
python setup.py install

./run_preprocessing.sh

./run_conversion.sh
```

### Build Docker Image

```
docker build --pull -t mlperf-lambda:transformer .
```

### Run Benchmark

```
DATADIR=/home/ubuntu/data/mlperf/transformer/examples/transformer/wmt14_en_de/utf8 PULL=0 LAMBDASYSTEM=Lambda_dual_basic ./run.sub
```
