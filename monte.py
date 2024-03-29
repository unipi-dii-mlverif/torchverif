import torch

class cubic(torch.nn.Module):
    def __init__(self):
        super(cubic, self).__init__()
        return

    def forward(self, x):
        return x ** 3

nn = torch.load("./examples/attack_nn_cubic.pth", map_location='cpu')
nn = torch.nn.Sequential(*(list(nn.children())[:-1]))

region = torch.FloatTensor(1000, 1, 1, 4).uniform_(-1, 1)

reach = nn(region)

print(torch.min(reach, 0))


print(nn(torch.Tensor([0.,0.,0.,0.])))