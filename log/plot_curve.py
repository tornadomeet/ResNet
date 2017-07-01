import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

res = [re.compile('.*Epoch\[(\d+)\] .*Train-accuracy.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Validation-accuracy.*=([.\d]+)')]


def plot_acc(log_name, color="r"):

    train_name = log_name.replace(".log", " train")
    val_name = log_name.replace(".log", " val")

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:  # i=0, match train acc
                break
            i += 1  # i=1, match validation acc
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])
        if epoch not in data:
            data[epoch] = [0] * len(res) * 2
        data[epoch][i*2] += val  # data[epoch], val:number
        data[epoch][i*2+1] += 1

    train_acc = []
    val_acc = []
    for k, v in data.items():
        if v[1]:
            train_acc.append(1.0 - v[0]/(v[1]))
        if v[2]:
            val_acc.append(1.0 - v[2]/(v[3]))

    x_train = np.arange(len(train_acc))
    x_val = np.arange(len(val_acc))
    plt.plot(x_train, train_acc, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.plot(x_val, val_acc, '-', linestyle='-', color=color, linewidth=2, label=val_name)
    plt.legend(loc="best")
    plt.xticks(np.arange(0, 131, 10))
    plt.yticks(np.arange(0.1, 0.71, 0.05))
    plt.xlim([0, 130])
    plt.ylim([0.1, 0.7])

def main():
    plt.figure(figsize=(14, 8))
    plt.xlabel("epoch")
    plt.ylabel("Top-1 error")
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    log_files = [i for i in args.logs.split(',')]
    color = color[:len(log_files)]
    for c in range(len(log_files)):
        plot_acc(log_files[c], color[c])
    plt.grid(True)
    plt.savefig(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves, using like: \n'
                                     'python -u plot_curve.py --log=resnet-18.log,resnet-50.log')
    parser.add_argument('--logs', type=str, default="resnet-50.log",
                        help='the path of log file, --logs=resnet-50.log,resnet-101.log')
    parser.add_argument('--out', type=str, default="training-curve.png", help='the name of output curve ')
    args = parser.parse_args()
    main()
