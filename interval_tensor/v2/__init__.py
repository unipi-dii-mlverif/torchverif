import functools
import torch
import numpy as np


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


HANDLED_FUNCTIONS = {}


class IntervalTensor(object):
    def __init__(self, inf, sup, gen = False):
        assert inf.shape == sup.shape
        assert torch.all(inf <= sup)
        self._inf = inf
        self._sup = sup
        self._gen = None
        if gen is True:
            self._gen = interval_from_infsup(inf, sup)

    def __repr__(self):
        return "interval_inf(value={}),\n interval_sup(value={}),".format(self._inf, self._sup)

    def __len__(self):
        return len(self._inf)

    def shape(self):
        return self._inf.shape

    def dim(self):
        return len(self._inf.shape)

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

    def __div__(self, other):
        if type(other) != IntervalTensor:
            m_inf = self._inf / other
            m_sup = self._sup / other
            return IntervalTensor(torch.minimum(m_inf, m_sup), torch.maximum(m_inf, m_sup))

        rec = IntervalTensor(torch.reciprocal(other._sup), torch.reciprocal(other._inf))
        return self*rec

    def flatten(self):
        finf = torch.flatten(self._inf)
        fsup = torch.flatten(self._sup)
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
    linf = torch.nn.functional.linear(input._inf, weight, bias)
    lsup = torch.nn.functional.linear(input._sup, weight, bias)
    return IntervalTensor(linf, lsup)


@implements(torch.nn.functional.conv2d)
def Conv2d(image, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    cinf = torch.nn.functional.conv2d(image._inf, weight, bias, stride, padding, dilation, groups)
    csup = torch.nn.functional.conv2d(image._sup, weight, bias, stride, padding, dilation, groups)
    return IntervalTensor(cinf, csup)


@implements(torch.nn.functional.max_pool2d)
def MaxPool2D(image, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, ceil_mode=False,
              return_indices=False):
    minf = torch.nn.functional.max_pool2d(image._inf, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    msup = torch.nn.functional.max_pool2d(image._sup, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return IntervalTensor(minf, msup)


@implements(torch.nn.functional.batch_norm)
def BatchNorm2D(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05,
                track_running_stats=True):
    binf = torch.nn.functional.batch_norm(input._inf, running_mean, running_var, weight, bias, training, momentum, eps)
    bsup = torch.nn.functional.batch_norm(input._sup, running_mean, running_var, weight, bias, training, momentum, eps)
    return IntervalTensor(binf, bsup)


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

def from_np_supinf(sup_arr, inf_arr):
    dst = torch.stack([inf_arr, sup_arr], dim=-1)
    tensor_int = IntervalTensor(dst.detach().numpy())
    return tensor_int


def interval_from_infsup(inf_arr, sup_arr, samples=1000):
    torch.manual_seed(9999)
    dist = torch.distributions.uniform.Uniform(inf_arr, sup_arr)
    return dist.sample([samples])


if __name__ == '__main__':
    _inf = torch.randn([4,3,224,224])
    _sup = _inf+1
    t1 = IntervalTensor(_inf, _sup)
    print(t1*2)
