import torch

from interval_tensor import *
import torch.nn as nn
import matplotlib.pyplot as plt


def test_batchn():
    model = torch.load("../models/conv_model.pth",map_location=torch.device('cpu'))

    for l in model:
        if isinstance(l,nn.BatchNorm2d):
            print(l.weight)
            print(l.running_mean)
            print(l.running_var)

def test_var():
    tensor_i = torch.randn((1,3,2,2))
    tensor_int = IntervalTensor(torch.unsqueeze(tensor_i,-1).numpy())

    print(torch.var(tensor_i, dim=(2,3)))
    print(torch.var(tensor_int))

    print(torch.mean(tensor_i, dim=(2,3)))
    print(torch.mean(tensor_int))

def testconvnet():
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


if __name__ == '__main__':
    test_var()