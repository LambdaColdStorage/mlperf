# Lambda Note


### Prepare Data

Step Zero: Create Python Virtual Env

```
cd ~
virtualenv -p /usr/bin/python3.6 venv-mxnet
. venv-mxnet/bin/activate
pip install opencv-python mxnet
```

Step one: Unzip ILSVRC2012_img_val.tar

```
mkdir -p ~/data/mlperf/imagenet-mxnet/val-jpeg
tar -vxf ILSVRC2012_img_val.tar -C ~/data/mlperf/imagenet-mxnet/val-jpeg

# https://github.com/NVIDIA/DeepLearningExamples/issues/144
cd ~/data/mlperf/imagenet-mxnet/val-jpeg
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

Step Two: Unzip ILSVRC2012_img_train.tar

```
mkdir -p ~/data/mlperf/imagenet-mxnet/train-jpeg
mkdir -p ~/data/mlperf/imagenet-mxnet/train-tar
tar -vxf ILSVRC2012_img_train.tar -C ~/data/mlperf/imagenet-mxnet/train-tar
./untar_imagenet_train.sh
```

Where `untar_imagenet_train.sh` is

```
#!/bin/bash
for filename in /home/ubuntu/data/mlperf/imagenet-mxnet/train-tar/*.tar; do
        #echo $filename
        #echo $(basename "$filename" .tar)
        outputname="/home/ubuntu/data/mlperf/imagenet-mxnet/train-jpeg/$(basename "$filename" .tar)"
        mkdir -p $outputname
        tar -vxf $filename -C $outputname
done
```

Step Four: Convert images to record files

```
python /home/ubuntu/venv-mxnet/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive train /home/ubuntu/data/mlperf/imagenet-mxnet/train-jpeg

python /home/ubuntu/venv-mxnet/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive val /home/ubuntu/data/mlperf/imagenet-mxnet/val-jpeg

python /home/ubuntu/venv-mxnet/lib/python3.6/site-packages/mxnet/tools/im2rec.py --pass-through --num-thread 20 train /home/ubuntu/data/mlperf/imagenet-mxnet/train-jpeg

python /home/ubuntu/venv-mxnet/lib/python3.6/site-packages/mxnet/tools/im2rec.py --pass-through --num-thread 20 val /home/ubuntu/data/mlperf/imagenet-mxnet/val-jpeg
```

### Build Docker Image

Change 'config-DGX1.sh' accordingly.

```
docker build --pull -t mlperf-nvidia:image_classification .
```

### Run Benchmark

```
DATADIR=/home/ubuntu LOGDIR=/home/ubuntu DGXSYSTEM=DGX1 ./run.sub
DATADIR=/home/ubuntu LOGDIR=/home/ubuntu DGXSYSTEM=Lambda_dual_basic ./run.sub
```

### Notes

Use `resnet-v1b-normconv-fl` for server. However this does not work with workstations (`<stderr>:This NormalizedConvolution is not supported by cudnn, MXNET NormalizedConvolution is applied.`) Use `resnet-v1b-fl` for workstation instead. 
