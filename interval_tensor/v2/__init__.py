import functools
import math

import torch
import numpy as np
import torch.nn.functional


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


HANDLED_FUNCTIONS = {}


class IntervalTensor(object):
    def __init__(self, inf: torch.Tensor, sup: torch.Tensor):
        if inf.shape != sup.shape:
            raise ValueError(f'Shape mismatch: {inf.shape}, {sup.shape}')
        self.shape = inf.shape
        self._inf = inf
        self._sup = sup

    def __repr__(self):
        return "interval_inf(value={}),\n interval_sup(value={}),".format(self._inf, self._sup)

    def __len__(self):
        return len(self._inf)

    def size(self, dim=None):
        return self._inf.size(self, dim)

    def _size(self):
        return np.prod(self._inf.shape)

    def dim(self):
        return len(self.shape)

    def __lt__(self, other):
        return (self._inf < other._inf) | ((self._inf == other._inf) & (self._sup < other._sup))

    def __gt__(self, other):
        return (self._inf > other._inf) | ((self._inf == other._inf) & (self._sup > other._sup))

    def __eq__(self, other):
        return (self._inf == other._inf) & (self._sup == other._sup)

    def check_interval(self):
        assert torch.all(self._inf <= self._sup)
        return torch.all(self._inf <= self._sup)

    def sample(self, samples=50):
        return interval_from_infsup(self._inf, self._sup, samples)

    def dim(self):
        return len(self._inf.shape)

    def __getitem__(self, item):
        return IntervalTensor(self._inf[item], self._sup[item])

    def __setitem__(self, key, value):
        self._inf[key] = value._inf
        self._sup[key] = value._sup

    def __add__(self, other):
        if type(other) != IntervalTensor:
            return IntervalTensor(self._inf + other, self._sup + other)

        return IntervalTensor(self._inf + other._inf, self._sup + other._sup)

    def __sub__(self, other):
        if type(other) != IntervalTensor:
            return IntervalTensor(self._inf - other, self._sup - other)

        return IntervalTensor(self._inf - other._sup, self._sup - other._inf)

    def __mul__(self, other):
        if type(other) != IntervalTensor:
            m_inf = self._inf * other
            m_sup = self._sup * other
            return IntervalTensor(torch.minimum(m_inf, m_sup), torch.maximum(m_inf, m_sup))

        m0 = self._inf * other._inf
        m1 = self._inf * other._sup
        m2 = self._sup * other._inf
        m3 = self._sup * other._sup

        return IntervalTensor(
            torch.minimum(torch.minimum(m0, m1), torch.minimum(m2, m3)),
            torch.maximum(torch.maximum(m0, m1), torch.maximum(m2, m3))
        )

    def __truediv__(self, other):
        if type(other) != IntervalTensor:
            m_inf = self._inf / other
            m_sup = self._sup / other
            return IntervalTensor(torch.minimum(m_inf, m_sup), torch.maximum(m_inf, m_sup))

        rec = IntervalTensor(torch.reciprocal(other._sup), torch.reciprocal(other._inf))
        return self * rec

    def flatten(self, start_dim=0, end_dim=-1):
        finf = torch.flatten(self._inf, start_dim, end_dim)
        fsup = torch.flatten(self._sup, start_dim, end_dim)
        return IntervalTensor(finf, fsup)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, IntervalTensor))
                for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


@implements(torch.nn.functional.linear)
def Linear(input, weight, bias=None):
    wshape = weight.shape
    output = IntervalTensor(torch.zeros(wshape[0]), torch.ones(wshape[0]))
    for i, w in enumerate(weight):
        output[i] = torch.sum(input * w)
    if bias is not None:
        output = output + bias
    return output


@implements(torch.nn.functional.conv2d)
def Conv2d(image, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    ishape = image.shape
    batches = ishape[0]
    print(ishape)
    wshape = weight.shape
    print(wshape)
    pshape = padding
    dshape = dilation
    hout = int((ishape[2] + 2 * pshape[0] - dshape[0] * (wshape[2] - 1) - 1) / stride[0]) + 1
    wout = int((ishape[3] + 2 * pshape[1] - dshape[1] * (wshape[3] - 1) - 1) / stride[1]) + 1
    cout = wshape[0]
    cin = wshape[1]
    fh = wshape[2]
    fw = wshape[3]

    hs = stride[0]
    ws = stride[1]

    output = IntervalTensor(torch.empty((batches, cout, hout, wout)), torch.empty((batches, cout, hout, wout)))
    print(output.shape)
    for batch in range(batches):
        for cout_j in range(cout):
            print("Conv2d", cout_j, cout, end="\r\n")
            conv_accum = IntervalTensor(torch.empty((hout, wout)), torch.empty((hout, wout)))

            kernels = weight[cout_j, :, :, :]
            img = image[batch, :, :, :]
            for i in range(hout):
                for j in range(wout):
                    conv_accum[i, j] = torch.sum(img[:, hs:hs + fh, ws:ws + fw] * kernels)

            if bias is not None:
                b = bias[cout_j]
                conv_accum += b

            output[batch, cout_j] = conv_accum
    return output


@implements(torch.nn.functional.max_pool2d)
def MaxPool2D(image, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, ceil_mode=False,
              return_indices=False):
    batches = image.shape[0]
    hout = int((image.shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    wout = int((image.shape[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    cout = image.shape[1]

    output = IntervalTensor(torch.empty((batches, cout, hout, wout)), torch.empty((batches, cout, hout, wout)))
    for batch in range(batches):
        for cout_j in range(cout):
            img = image[batch, cout_j]
            pool = IntervalTensor(torch.empty((hout, wout)), torch.empty((hout, wout)))
            for i in range(hout):
                for j in range(wout):
                    pool[i, j] = torch.max(img[i:i + kernel_size, j:j + kernel_size])
            output[batch, cout_j] = pool
    return output


@implements(torch.nn.functional.batch_norm)
def BatchNorm2D(input, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05,
                track_running_stats=True):
    ishape = input.shape
    h = ishape[2]
    w = ishape[3]
    cout = ishape[1]
    batches = input.shape[0]
    output = IntervalTensor(torch.empty((batches, cout, h, w)), torch.empty((batches, cout, h, w)))
    for cout_j in range(cout):

        img = input[:, cout_j, :, :]
        mean = torch.mean(img)
        var = torch.var(img)
        if track_running_stats and running_mean is not None and running_var is not None:
            mean = mean * (momentum) + (1 - momentum) * running_mean[cout_j]
            var = var * (momentum) + (1 - momentum) * running_var[cout_j]

        bi = bias[cout_j]
        we = weight[cout_j]
        norm_img = (img - mean) / torch.sqrt(var + eps)
        output[:, cout_j] = norm_img * we + bi

    return output


@implements(torch.cat)
def Cat(tensors, dim=0, out=None):
    cat_inf = None
    cat_sup = None
    for t in tensors:
        if cat_inf is None:
            cat_inf = t._inf
        else:
            cat_inf = torch.cat((cat_inf, t._inf))
        if cat_sup is None:
            cat_sup = t._sup
        else:
            cat_sup = torch.cat((cat_sup, t._sup))

    return IntervalTensor(cat_inf, cat_sup)


@implements(torch.sum)
def Sum(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        sinf = torch.sum(input._inf)
        ssup = torch.sum(input._sup)
    else:
        sinf = torch.sum(input._inf, dim, keepdim, dtype)
        ssup = torch.sum(input._sup, dim, keepdim, dtype)
    return IntervalTensor(sinf, ssup)


@implements(torch.squeeze)
def Squeeze(input, dim):
    return IntervalTensor(torch.squeeze(input._inf, dim), torch.squeeze(input._sup, dim))


@implements(torch.max)
def Max(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        fl = input._inf.flatten()
        su = input._sup.flatten()
        v = torch.max(fl)
        i = torch.argmax(fl)
        return IntervalTensor(v, su[i])
    i = torch.argmax(input._inf, dim, keepdim=True)
    res = IntervalTensor(torch.gather(input._inf, dim, i), torch.gather(input._sup, dim, i))
    if not keepdim:
        return torch.squeeze(res, dim=dim)
    return res


@implements(torch.min)
def Min(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        fl = input._inf.flatten()
        su = input._sup.flatten()
        v = torch.min(fl)
        i = torch.argmin(fl)
        return IntervalTensor(v, su[i])
    i = torch.argmin(input._inf, dim, keepdim=True)
    res = IntervalTensor(torch.gather(input._inf, dim, i), torch.gather(input._sup, dim, i))
    if not keepdim:
        return torch.squeeze(res, dim=dim)
    return res


@implements(torch.mean)
def Mean(input):
    return IntervalTensor(torch.mean(input._inf), torch.mean(input._sup))


@implements(torch.var)
def Var(input, correction=1):
    im = torch.square(input - torch.mean(input))
    var = torch.sum(im) * (1 / (im._size() - correction))
    return var


@implements(torch.nn.functional.relu)
def ReLU(input, inplace=False):
    rinf = torch.nn.functional.relu(input._inf)
    rsup = torch.nn.functional.relu(input._sup)
    return IntervalTensor(rinf, rsup)


@implements(torch.sigmoid)
def Sigmoid(input, inplace=False):
    rinf = torch.nn.functional.sigmoid(input._inf)
    rsup = torch.nn.functional.sigmoid(input._sup)
    return IntervalTensor(rinf, rsup)


@implements(torch.tanh)
def Tanh(input, inplace=False):
    rinf = torch.nn.functional.tanh(input._inf)
    rsup = torch.nn.functional.tanh(input._sup)
    return IntervalTensor(rinf, rsup)

@implements(torch.sqrt)
def Sqrt(input, inplace=False):
    rinf = torch.sqrt(input._inf)
    rsup = torch.sqrt(input._sup)
    return IntervalTensor(rinf, rsup)

@implements(torch.square)
def Square(input, inplace=False):
    rinf = torch.square(input._inf)
    rsup = torch.square(input._sup)
    return IntervalTensor(rinf, rsup)


def from_np_supinf(sup_arr, inf_arr):
    dst = torch.stack([inf_arr, sup_arr], dim=-1)
    tensor_int = IntervalTensor(dst.detach().numpy())
    return tensor_int


def interval_from_infsup(inf_arr, sup_arr, samples=50):
    torch.manual_seed(9999)
    dist = torch.distributions.uniform.Uniform(inf_arr, sup_arr)
    return dist.sample([samples])


if __name__ == '__main__':
    _inf = torch.randn([2, 3, 32, 32])
    _sup = _inf
    t1 = IntervalTensor(_inf, _inf)
    fil = torch.randn([3,3,3,3])
    m = torch.Tensor([0,0,0])
    v = torch.Tensor([1,1,1])
    a = torch.nn.functional.conv2d(_inf, fil, stride=[2,2], padding=[0,0])
    b = torch.nn.functional.conv2d(t1, fil, stride=[2,2], padding=[0,0])


