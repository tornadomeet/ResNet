# training log description

## Imagenet 1K classes

| network    | gpus | batch-size | mxnet version | description |
| :---------:| :---:| :---------:|:----: | :-----------:|
| resnet-18 |  2 x k80  | 512  |[73a0f6e](https://github.com/dmlc/mxnet/commit/73a0f6eb7f5570c3a8aa93f9e1fa6bf257a7bdd8) |learning rate decrease by 0.1 at [60, 75, 90] epoch, centos7 + openblas + cuda7.5+cudnn v5/v5.1|
| resnet-50 |  3 x k80  | 256  |[73a0f6e](https://github.com/dmlc/mxnet/commit/73a0f6eb7f5570c3a8aa93f9e1fa6bf257a7bdd8) |learning rate decrease by 0.1 at [30, 60, 90] epoch, centos7 + openblas + cuda7.5+cudnn v5/v5.1|
| resnet-152 | 8 x m40 | 256 | [73a0f6e](https://github.com/dmlc/mxnet/commit/73a0f6eb7f5570c3a8aa93f9e1fa6bf257a7bdd8) | others are same as above, but disable the recommend aug since epoch 101 |

## Cifar10

| network    | gpus | batch-size | mxnet version | description |
| :---------:| :---:| :---------:|:-------------:|:-----------:|
|  resnet-164|  2 gtx 1080   | 128        | [73a0f6e](https://github.com/dmlc/mxnet/commit/73a0f6eb7f5570c3a8aa93f9e1fa6bf257a7bdd8)| learning rate decrease by 0.1 at [220, 260, 280] epoch, cuda8.0 + cudnn v5.1|
