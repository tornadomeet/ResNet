training log description
=====================================
| data-type     | network    | gpus | batch-size | description |
| :------------ | :---------:| :---:| :---------:|:-----------:|
| imagenet      |  resnet-50 |  6   | 256        |k80, learning rate decrease by 0.1 at [30, 60, 90] epoch, centos7 + openblas + cuda7.5+cudnn v5/v5.1|
| cifar10       |  resnet-164|  2   | 128        | gtx 1080, learning rate decrease by 0.1 at [220, 260, 280] epoch, cuda8.0 + cudnn v5.1|
