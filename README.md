# Reproduce ResNet-v2 using MXNet


------------------------------------------
##### How to Train
######cifar10
first you should use ```im2rec``` to create the .rec file, then training with cmd like this:
```shell
python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 --batch-size 128 --num-examples 50000 --gpus=4,5,6,7
```
change ```depth``` when training different model, only support```(depth-2)%9==0```, such as RestNet-110, ResNet-164, ResNet-1001...

######imanget
same as training cifar10, you should create .rec file first, i recommend use this cmd parameters:  
```shell
$im2rec_path train_cls_480.lst train/ mxnet/train_480.rec resize=480
```
set ```resiet=480``` here may use more disk memory, but this is can be done with scale augmentation during training[1][2].

because you are training imagnet this time, so you should change ```data-type = imagenet```, training cmd is like this:
```shell
python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 258 --num-examples 1281167 --gpus=2,3,4,5,6,7
```
change depth to different number to support different model, currently suport ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNet-200.

######retrain
When training large dataset(like imagnet), it's better for us to change learning rate manually, so retrain is very important.   
the code here support retrain, suppose you want to retrain your resnet-50 model from epoch 70 and want to change lr=0.0005, wd=0.001, batch-size=156 using 8gpu, then you can try this cmd:
```shell
python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 256 \
--num-examples 1281167 --gpus=0,1,2,3,4,5,6,7 --model-load-epoch=70 --lr 0.0005 --wd 0.001 --retrain
```  

----------------------------------
#####Result
######cifar 10
1-crop validation error on cifar10 (32*32):
| Network    | top-1 |
| :------:   | :---: |
|ResNet-164  | 5.46% |

----------------------------------------
######imagenet
1-crop validation error on imagenet (center 224x224 crop from resized image with shorter side=256):
| model   | top-1 | top-5 |
|:------: |:-----:|:-----:|
|ResNet-50|25.45% |  7.96%|

----------------------------------------
#####Notes
* I trained ResNet-50, but the result is not so good, top-1/top-5 acc of ResNet-50 is about 0.75%/0.16% worse than [ResNet-v1 caffe result](https://github.com/KaimingHe/deep-residual-networks/blob/master/README.md#results), this may due to io implentation in MXNet is some different(https://github.com/dmlc/mxnet/issues/2944).
* Consider using torch io instead of mxnet io for training in the future.

#####Reference
[1] He, Kaiming, et al. "Deep Residual Learning for Image Recognition." arXiv arXiv:1512.03385 (2015).  
[2] He, Kaiming, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027 (2016)
