import numpy as np
import cv2
import mxnet as mx
import argparse

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def main():
    synset = [l.strip() for l in open(args.synset).readlines()]
    img = cv2.imread(args.img)  # read image in b,g,r order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to r,g,b order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)

    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    arg_params["data"] = mx.nd.array(img, ctx)
    arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
    exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)

    prob = np.squeeze(exe.outputs[0].asnumpy())
    pred = np.argsort(prob)[::-1]
    print("Top1 result is: ", synset[pred[0]])
    print("Top5 result is: ", [synset[pred[i]] for i in range(5)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use pre-trainned resnet model to classify one image")
    parser.add_argument('--img', type=str, default='test.jpg', help='input image for classification')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--synset', type=str, default='synset.txt', help='file mapping class id to class name')
    parser.add_argument('--prefix', type=str, default='resnet-50', help='the prefix of the pre-trained model')
    parser.add_argument('--epoch', type=int, default=0, help='the epoch of the pre-trained model')
    args = parser.parse_args()
    main()