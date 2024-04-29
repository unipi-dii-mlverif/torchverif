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
        if len(intervals) > 0:
            self._value = IntervalTensor.from_np(intervals)

    def __repr__(self):
        return "interval_tensor(value={})".format(self._value)

    def data(self):
        return self._value

    def from_raw(np_intervals):
        i = IntervalTensor(np_intervals)
        i._value = np_intervals
        return i


    def shape(self):
        return self._value.shape

    def sample(self, num_samples):
        as_list = self._value.flatten()
        samples = []
        for i in as_list:
            sam = torch.distributions.uniform.Uniform(i[0].inf, i[0].sup).sample([num_samples])
            samples.append(sam)
        stacked = torch.stack(samples, 1)
        return stacked.reshape((num_samples, *self.shape()))

    def inf(self):
        as_list = self._value.flatten()
        inf_list = []
        for e in as_list:
            inf_list.append(e[0].inf)
        return np.array(inf_list).reshape(self._value.shape)

    def sup(self):
        as_list = self._value.flatten()
        inf_list = []
        for e in as_list:
            inf_list.append(e[0].sup)
        return np.array(inf_list).reshape(self._value.shape)

    def from_np(np_array: np.ndarray):
        dim = list(np_array.shape)
        new_arr = np.empty(dim[:-1], dtype=object)
        area = new_arr.size
        new_arr = new_arr.reshape(area)
        if area == np_array.size:
            np_array_ = np_array.reshape(area, 1)
        else:
            np_array_ = np_array.reshape(area, 2)
        for i, pair in enumerate(np_array_):
            if pair is None:
                new_arr[i] = interval(0)
            elif len(pair) > 2:
                new_arr[i] = interval(*pair)
            else:
                new_arr[i] = interval(pair)
        new_arr = new_arr.reshape(dim[:-1])
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
    dim = len(weight)
    result = np.empty(dim, dtype=object)
    rint = IntervalTensor([])
    for h, r in enumerate(weight):
        r_accum = interval(0)
        for i, c in enumerate(r):
            r_accum += c.item() * input._value[i]
        if bias is not None and bias:
            r_accum += bias[h].item()
        result[h] = r_accum
    rint._value = result
    return rint


@implements(torch.nn.functional.conv2d)
def Conv2d(image, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ishape = image.shape()
    #print("Image size: ", image.shape())
    wshape = weight.shape
    #print("Parameter size: ", weight.shape)
    pshape = padding
    dshape = dilation

    hout = int((ishape[1] + 2 * pshape[0] - dshape[0] * (wshape[2] - 1) - 1) / stride[0]) + 1
    wout = int((ishape[2] + 2 * pshape[1] - dshape[1] * (wshape[3] - 1) - 1) / stride[1]) + 1
    cout = wshape[0]
    cin = wshape[1]
    fh = wshape[2]
    fw = wshape[3]
    #print("Output size: ", (cout, hout, wout))

    output = np.empty((cout, hout, wout), dtype=object)
    # Iterate over output channels
    for cout_j in range(cout):
        # Iterate over input channels
        conv_accum = IntervalTensor(np.zeros((hout, wout, 1)))
        _ca = conv_accum.data()
        #print(_ca.shape)
        for k in range(cin):
            #print("Out channel: ", cout_j, " In channel: ", k)

            # Collect elements for convolution
            # kernel (cout_j, k, :, :) and
            # image (k, :, :)
            kernel = weight[cout_j, k, :, :]
            img = image.data()[k, :, :]

            # Perform convolution
            # between kernel and img

            for i in range(hout):
                for j in range(wout):
                    accum = interval(0)
                    for f in range(fh):
                        for g in range(fw):
                            accum += kernel[f, g] * img[i + f, j + g]

                    _ca[i, j] += accum + bias[cout_j]
        output[cout_j] = _ca
    #print(output)
    return IntervalTensor.from_raw(output)


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
