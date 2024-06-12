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

    def __len__(self):
        return len(self._value)

    def data(self):
        return self._value

    def from_raw(np_intervals):
        i = IntervalTensor([])
        i._value = np_intervals
        return i

    def shape(self):
        return self._value.shape

    def dim(self):
        return len(self._value.shape)

    def flatten(self, start, end):
        return IntervalTensor.from_raw(np.ndarray.flatten(self._value))

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
        if bias is not None:
            r_accum += bias[h].item()
        result[h] = r_accum
    rint._value = result
    return rint


@implements(torch.nn.functional.conv2d)
def Conv2d(image, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ishape = image.shape()
    wshape = weight.shape

    pshape = padding
    dshape = dilation

    hout = int((ishape[2] + 2 * pshape[0] - dshape[0] * (wshape[2] - 1) - 1) / stride[0]) + 1
    wout = int((ishape[3] + 2 * pshape[1] - dshape[1] * (wshape[3] - 1) - 1) / stride[1]) + 1
    cout = wshape[0]
    cin = wshape[1]
    fh = wshape[2]
    fw = wshape[3]

    output = np.empty((1, cout, hout, wout), dtype=object)
    # Iterate over output channels
    for cout_j in range(cout):
        # Iterate over input channels
        conv_accum = IntervalTensor(np.zeros((hout, wout, 1)))
        _ca = conv_accum.data()

        for k in range(cin):
            print(f'[DEBUG] INTERVAL CONV {cout_j},{k}')
            # Collect elements for convolution
            # kernel (cout_j, k, :, :) and
            # image (k, :, :)
            kernel = weight[cout_j, k, :, :]
            img = image.data()[0, k, :, :]

            # Perform convolution
            # between kernel and img

            for i in range(hout):
                for j in range(wout):
                    accum = interval(0)
                    for f in range(fh):
                        for g in range(fw):
                            accum += kernel[f, g] * img[i + f, j + g]

                    _ca[i, j] += accum
        # Apply bias
        if bias is not None:
            b = bias[cout_j]
            for i in range(hout):
                for j in range(wout):
                    _ca[i, j] += b

        output[0, cout_j] = _ca

    return IntervalTensor.from_raw(output)


@implements(torch.nn.functional.max_pool2d)
def MaxPool2D(image, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, ceil_mode=False,
              return_indices=False):
    ishape = image.shape()

    wshape = kernel_size
    pshape = padding
    dshape = dilation

    hout = int((ishape[2] + 2 * pshape - dshape * (wshape - 1) - 1) / stride) + 1
    wout = int((ishape[3] + 2 * pshape - dshape * (wshape - 1) - 1) / stride) + 1
    cout = ishape[1]
    fh = wshape
    fw = wshape

    output = np.empty((1, cout, hout, wout), dtype=object)
    # Iterate over output channels
    for cout_j in range(cout):
        # Iterate over input channels
        max_pool = IntervalTensor(np.zeros((hout, wout, 1)))
        _ca = max_pool.data()

        # Collect elements for convolution
        # kernel (cout_j, k, :, :) and
        # image (k, :, :)
        img = image.data()[0, cout_j, :, :]

        # Perform convolution
        # between kernel and img

        for i in range(hout):
            for j in range(wout):
                accum = interval(img[i, j])
                for f in range(fh):
                    for g in range(fw):
                        accum = max(accum, img[i + f, j + g])

                _ca[i, j] = accum
        output[0, cout_j] = _ca

    return IntervalTensor.from_raw(output)


@implements(torch.nn.functional.batch_norm)
def BatchNorm2D(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05,
                track_running_stats=True):
    ishape = input.shape()
    h = ishape[2]
    w = ishape[3]
    cout = ishape[1]

    # Compute channel-wise mean and variance
    ch_var = torch.var(input, correction=0)
    ch_mean = torch.mean(input)

    output = np.empty((1, cout, h, w), dtype=object)
    for cout_j in range(cout):
        # Iterate over output channels
        batch_norm = IntervalTensor(np.zeros((h, w, 1)))
        _ca = batch_norm.data()

        # Collect elements for normalization
        # image (k, :, :)
        # mean  (k, :, :)
        # var   (k, :, :)
        img = input.data()[0, cout_j, :, :]
        mean = (ch_mean.data()[cout_j])

        var = ch_var.data()[cout_j]

        if track_running_stats and running_mean is not None and running_var is not None:
            mean = mean * (momentum) + (1 - momentum) * running_mean[cout_j]
            var = var * (momentum) + (1 - momentum) * running_var[cout_j]

        bi = bias[cout_j]
        we = weight[cout_j]
        for i in range(h):
            for j in range(w):
                a = img[i, j]
                a = (a - mean) / imath.sqrt(var + eps)
                _ca[i, j] = a * we + bi

        output[0, cout_j] = _ca

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


@implements(torch.var)
def Var(input, dim=None, keepdim=False, correction=1):
    # Simplified version for channel-wise variance on 3D tensors with batch_size=1
    batch_size = input.shape()[0]
    assert batch_size == 1, "Not implemented for batch_size > 1 "
    cout = input.shape()[1]
    hin = input.shape()[2]
    win = input.shape()[3]
    n = (hin * win) - correction
    # iterate over channels to compute variance
    output = np.empty((cout), dtype=object)
    for cout_j in range(cout):
        # Collect channel tensor to compute variance
        img = input.data()[0, cout_j, :, :]
        # First compute mean
        img_mean = interval(0)
        for h in range(hin):
            for w in range(win):
                img_mean = img_mean + img[h, w]
        img_mean /= win * hin

        # Then compute variance
        r_accum = interval(0)
        for h in range(hin):
            for w in range(win):
                r_accum = r_accum + (img[h, w] - img_mean) ** 2
        output[cout_j] = r_accum / n
    return IntervalTensor.from_raw(output)


@implements(torch.mean)
def Mean(input):
    # Simplified version for channel-wise variance on 3D tensors with batch_size=1
    batch_size = input.shape()[0]
    assert batch_size == 1, "Not implemented for batch_size > 1 "
    cout = input.shape()[1]
    hin = input.shape()[2]
    win = input.shape()[3]
    n = (hin * win)
    # iterate over channels to compute variance
    output = np.empty((cout), dtype=object)
    for cout_j in range(cout):
        # Collect channel tensor to compute variance
        img = input.data()[0, cout_j, :, :]
        # First compute mean
        img_mean = interval(0)
        for h in range(hin):
            for w in range(win):
                img_mean = img_mean + img[h, w]

        output[cout_j] = img_mean / n
    return IntervalTensor.from_raw(output)


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


def from_np_supinf(sup_arr, inf_arr):
    dst = torch.stack([inf_arr, sup_arr], dim=-1)
    tensor_int = IntervalTensor(dst.numpy())
    return tensor_int


def interval_from_supinf(sup_arr, inf_arr, samples=100):
    dist = torch.distributions.uniform.Uniform(inf_arr, sup_arr)
    return dist.sample([samples])

