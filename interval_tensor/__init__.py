import functools
import torch
from interval import interval, imath
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
    def __init__(self, intervals):
        self._value = IntervalTensor.from_np(intervals)

    def __repr__(self):
        return "interval_tensor(value={})".format(self._value)

    def data(self):
        return self._value

    def from_np(np_array):
        new_arr = []
        for pair in np_array:
            if len(pair) > 2:
                new_arr.append(interval(*pair))
            else:
                new_arr.append(interval(pair))

        return new_arr

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
    result = [interval(0)] * len(weight)
    rint = IntervalTensor(result)
    for h, r in enumerate(weight):
        r_accum = interval(0)
        for i, c in enumerate(r):
            r_accum += c.item() * input._value[i]
        r_accum += bias[h].item()
        result[h] = r_accum
    rint._value = result = result;
    return rint


@implements(torch.nn.functional.relu)
def ReLU(input, inplace=False):
    ints = input._value
    for i, v in enumerate(ints):
        ints[i] = relu_interval(v)
    return input


@implements(torch.sigmoid)
def Sigmoid(input, inplace=False):
    ints = input._value
    for i, v in enumerate(ints):
        ints[i] = sigmoid_interval(v)
    return input


@implements(torch.tanh)
def Tanh(input, inplace=False):
    ints = input._value
    for i, v in enumerate(ints):
        ints[i] = tanh_interval(v)
    return input


def sigmoid_interval(intr):
    return 1 / (1 + imath.exp(-intr))


def tanh_interval(intr):
    return imath.tanh(intr)


@interval.function
def relu_interval(c):
    lb = c.inf if c.inf > 0 else 0
    ub = c.sup if c.sup > 0 else 0
    return [[lb, ub]]

def extract_feature_tensor_bounds(feature_tensor):
    idata = feature_tensor.data()
    out_data = [np.array([])] * len(idata)
    for l_, i_ in enumerate(idata):
        feat_ = np.array(i_[0]).reshape(2, 1)
        label_ = (np.ones(2) * l_).reshape(2, 1)
        stacked = np.hstack((label_, feat_))
        out_data[l_] = stacked
    return out_data