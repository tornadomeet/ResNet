# training log description

## Imagenet 10K classes

| network    | gpus | batch-size | description |
| :---------:| :---:| :---------:|:-----------:|
| resnet-50 |  3 x k80  | 256  | learning rate decrease by 0.1 at [30, 60, 90] epoch, centos7 + openblas + cuda7.5+cudnn v5/v5.1|
| resnet-152 | 8 x m40 | 256 | m40, others are same as above, but disable the recommend aug since epoch 101 |

## Cifar10

| network    | gpus | batch-size | description |
| :---------:| :---:| :---------:|:-----------:|
|  resnet-164|  2   | 128        | gtx 1080, learning rate decrease by 0.1 at [220, 260, 280] epoch, cuda8.0 + cudnn v5.1|
