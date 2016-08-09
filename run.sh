#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=4,5,6,7

## train resnet-imagenet-50 
#python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 258 --num-examples 1281167 --gpus=2,3,4,5,6,7 --lr=0.1
