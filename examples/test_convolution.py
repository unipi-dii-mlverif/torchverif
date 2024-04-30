import torch

from interval_tensor import *
import torch.nn as nn
import matplotlib.pyplot as plt


def test_maxpool():
    tensor_i = torch.randn((1, 3, 5, 5))
    net = torch.nn.Sequential(
        torch.nn.MaxPool2d(5, 1, padding=0)
    )

    tensor_int = IntervalTensor(torch.unsqueeze(tensor_i, -1).numpy())
    print(tensor_i)
    max_i = net(tensor_i)
    max_int = net(tensor_int)
    print(max_i, max_int)


def test_batchn():
    tensor_i = torch.randn((1, 3, 2, 2))
    net = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3, track_running_stats=False),
        torch.nn.Flatten()
    )

    print(net[0].running_mean)
    print(net[0].running_var)

    tensor_int = IntervalTensor(torch.unsqueeze(tensor_i, -1).numpy())
    batch_i = net(tensor_i)
    batch_int = net(tensor_int)

    print(batch_i, "\n", batch_int)


def test_var():
    tensor_i = torch.randn((1, 3, 2, 2))
    tensor_int = IntervalTensor(torch.unsqueeze(tensor_i, -1).numpy())

    print(torch.var(tensor_i, dim=(2, 3)))
    print(torch.var(tensor_int))

    print(torch.mean(tensor_i, dim=(2, 3)))
    print(torch.mean(tensor_int))


def testconvnet():
    torch.manual_seed(9999)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 5, kernel_size=2, stride=1, bias=True)
    )

    inimg = torch.ones(1, 3, 3, 3)

    intimg = np.empty((1, 3, 3, 3, 2), dtype=object)

    for i in range(3):
        for j in range(3):
            intimg[0, 0, i, j] = [1, 1]
            intimg[0, 1, i, j] = [1, 1]
            intimg[0, 2, i, j] = [1, 1]

    intimg = IntervalTensor(intimg)
    print(inimg)

    output = net(inimg)
    intout = net(intimg)

    print(output[0][0])
    print(intout.data()[0][0])


if __name__ == '__main__':
    test_batchn()
