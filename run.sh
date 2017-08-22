#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
# python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 \
#       --batch-size 128 --num-classes 10 --num-examples 50000 --gpus=0,1
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 \
#      --batch-size 128 --num-classes 10 --num-examples 50000 --gpus=0,1,2,3,4,5,6,7

## train resnet-50
python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 \
       --batch-size 256 --gpus=0,1,2,3
