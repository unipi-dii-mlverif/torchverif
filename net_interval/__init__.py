from interval_tensor import IntervalTensor
from interval_tensor import extract_feature_tensor_bounds
import numpy as np
import torch
import matplotlib.pyplot as plt


def _get_output_shape(net):
    i = torch.Tensor(net[0].weight.shape)
    o = net(i)
    return o.shape[1]


def evaluate_net_interval(net, regions):
    out_classes = _get_output_shape(net)
    intervals = np.array([*regions], dtype=object)
    i = IntervalTensor(intervals)
    o = net(i)
    return o,np.array(extract_feature_tensor_bounds(o)).reshape(out_classes * 2, 2)


def evaluate_net_samples(net, regions, cartesian=True, samples=10):
    out_classes = _get_output_shape(net)
    features = []
    for r in regions:
        f = torch.distributions.uniform.Uniform(r[0], r[1]).sample([samples])
        features.append(f)
    stacked = None
    if cartesian:
        stacked = torch.cartesian_prod(*features)
    else:
        stacked = torch.stack(features, 1)
    sample_no = len(stacked)
    points = net(stacked).detach().numpy()
    poly = []
    for i in range(out_classes):
        poly_points = np.reshape(points[:, i], (sample_no, 1))
        poly_p_i = np.hstack((i * np.ones((len(poly_points), 1)), poly_points))
        poly.append(poly_p_i)
    return poly


def interval_plot_scores_helper(sample_group, bounds, threshold):
    for sample in sample_group:
        plt.scatter(sample[:, 1], sample[:, 0], s=10, marker=".")
    bbs = []
    for b in bounds:
        if len(bbs) < b[0]+1:
            bbs.append([])
        bbs[int(b[0])].append(b[1])

    for i,b in enumerate(bbs):
        plt.hlines(y=i, xmin=b[0], xmax=b[1], color="blue", linestyles="-.")
    plt.scatter(bounds[:, 1], bounds[:, 0], color="blue", marker="*");
    plt.axvline(threshold, color="green", linestyle="--")
    plt.ylabel("Label")
    plt.xlabel("Score")
    plt.yticks(bounds[:, 0])
    plt.show()

def verify_bound_disjunction(t_interval, check_class):
    output_intervals = t_interval.data()
    class_score_interval = output_intervals[check_class]
    intersections = 0
    for i,interval in enumerate(output_intervals):
        if i != check_class:
            intersect = interval & class_score_interval
            if len(intersect) > 0:
                intersections += 1
    return intersections == 0