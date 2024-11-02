import torch

from torchverif.interval_tensor.v2 import *
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


def test_batchn1d():
    tensor_i = torch.randn((1, 20))
    net = torch.nn.Sequential(
        torch.nn.BatchNorm1d(20),
    )

    net.eval()

    tensor_int = IntervalTensor(tensor_i, tensor_i)
    batch_i = net(tensor_i)
    batch_int = net(tensor_int)

    print(batch_i, "\n", batch_int)



def test_batchn():
    tensor_i = torch.randn((1, 3, 2, 2))
    net = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3, track_running_stats=False),
        torch.nn.Flatten() # Flatten
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
    net = torch.load("../models/conv_model_5ch.pth", map_location=torch.device('cpu'))
    net
    inimg = torch.randn(1, 3, 32, 32)
    tensor_int = IntervalTensor(inimg, inimg)

    intout = net(tensor_int)
    print(intout)


if __name__ == '__main__':
    test_batchn1d()
