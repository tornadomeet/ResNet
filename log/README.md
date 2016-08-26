training log description
=====================================
  | data-type     | network    | gpus | batch-size | description |
  | :------------ | :---------:| :---:| :---------:|:-----------:|
  | imagenet      |  Resnet-50 |  6   | 256        |k80, learning rate decrease by 0.1 at [30, 60, 90] epoch, centos7 + openblas + cuda7.5+cudnn v5/v5.1|
