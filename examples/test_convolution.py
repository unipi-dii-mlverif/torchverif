import torch

from interval_tensor import *
import torch.nn as nn
import matplotlib.pyplot as plt


if __name__ == '__main__':
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, kernel_size=2, stride=1)
    )

    inimg = torch.ones(3, 3, 3)

    intimg = np.empty((3, 3, 3, 2), dtype=object)

    for i in range(3):
        for j in range(3):
            intimg[0, i, j] = [-1,1]
            intimg[1, i, j] = [-1,1]
            intimg[2, i, j] = [-1,1]


    intimg = IntervalTensor(intimg)

    samples = intimg.sample(100)
    output = net(samples)
    intout = net(intimg)

    print(output.shape)
    print(intout.shape())
