import numpy as np
import torch
from torchverif.interval_tensor.v1 import IntervalTensor
from torchverif.interval_tensor.v1 import extract_feature_tensor_bounds


def _get_output_shape(net):
    i = torch.Tensor(net[0].weight.shape)
    o = net(i)
    return o.shape[1]


def evaluate_fcnn_interval(net, regions):
    out_classes = _get_output_shape(net)
    intervals = np.array([*regions], dtype=object)
    i = IntervalTensor(intervals)
    o = net(i)
    return o, np.array(extract_feature_tensor_bounds(o)).reshape(out_classes * 2, 2)



def evaluate_fcnn_samples(net, regions, cartesian=True, samples=10):
    out_classes = _get_output_shape(net)
    features = []
    for r in regions:
        if len(r) > 1:
            f = torch.distributions.uniform.Uniform(r[0], r[1]).sample([samples])
        else:
            f = r
        features.append(f)
    stacked = None
    if len(features[0]) > 1:
        if cartesian:
            stacked = torch.cartesian_prod(*features)
        else:
            stacked = torch.stack(features, 1)
    else:
        stacked = torch.Tensor(features).reshape(1, len(features))
    sample_no = len(stacked)
    points = net(stacked).detach().numpy()
    poly = []
    for i in range(out_classes):
        poly_points = np.reshape(points[:, i], (sample_no, 1))
        poly_p_i = np.hstack((i * np.ones((len(poly_points), 1)), poly_points))
        poly.append(poly_p_i)
    return poly




def verify_bound_disjunction(t_interval, check_class):
    output_intervals = t_interval.data()
    class_score_interval = output_intervals[check_class]
    intersections = 0
    for i, interval in enumerate(output_intervals):
        if i != check_class:
            intersect = interval & class_score_interval
            if len(intersect) > 0:
                intersections += 1
    return intersections == 0
