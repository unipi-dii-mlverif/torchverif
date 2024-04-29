from matplotlib import patches
from webencodings import labels

from interval_tensor import IntervalTensor
from interval_tensor import extract_feature_tensor_bounds
import numpy as np
import torch
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def evaluate_conv_samples(net, image, samples=10):
    out_classes = _get_output_shape(net)
    print(out_classes)


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


def interval_plot_scores_helper(sample_group, bounds, threshold=None, legend=None, class_labels=None, xticks=[], xlabel=None, ylabel=None, title=None):
    colors = ["darkorange", "royalblue", "lime"]
    for i, sample in enumerate(sample_group):
        l = plt.scatter(sample[:, 1], sample[:, 0], s=0.1, marker=".", color=colors[i])
        if legend is not None and legend > 1:
            l.set_label(class_labels[i])
    bbs = []
    for b in bounds:
        if len(bbs) < b[0] + 1:
            bbs.append([])
        bbs[int(b[0])].append(b[1])

    for i, b in enumerate(bbs):
        l = plt.scatter(y=[i, i], x=[bounds[2 * i, 1], bounds[2 * i + 1, 1]], color=colors[i], marker="*")
        if len(sample_group) < 1:
            l = plt.hlines(y=i, xmin=b[0], xmax=b[1], color=colors[i], linestyles="--")
        if legend is not None and legend >= 1:
            l.set_label(class_labels[i])

    if threshold is not None:
        plt.axvline(threshold, color="green", linestyle="--")
    plt.ylabel("Output neuron")
    plt.xlabel("Prediction")
    plt.yticks(bounds[:, 0])
    if legend is not None and legend > 0:
        plt.legend()
    plt.title(title if title is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.xlabel(xlabel if xlabel is not None else "")


def interval_time_plot_helper(bound_list, neuron=None, class_labels=None, xticks=[], xlabel=None, ylabel=None, title=None):
    colors = ["darkorange", "royalblue", "lime"]
    hatches = ["O", "//", "xx", "\\\\", "//"]
    class_number = int(len(bound_list[0]) / 2)
    print(class_number)
    time_series = np.empty([len(bound_list[0]), len(bound_list)])  # as the number of output labels*2 (up and inf)
    timestamps = np.linspace(0, len(bound_list), len(bound_list))
    fig, ax = plt.subplots(1,1)
    for time, bounds in enumerate(bound_list):
        bbs = []
        for b in bounds:
            if len(bbs) < b[0] + 1:
                bbs.append([])
            bbs[int(b[0])].append(b[1])

        for i, b in enumerate(bbs):
            time_series[2 * i, time] = bounds[2 * i, 1]
            time_series[2 * i + 1, time] = bounds[2 * i + 1, 1]
    legend_handles = []
    for i in range(class_number):
        color_idx = i
        print(color_idx)
        if (neuron is not None and color_idx == neuron) or neuron is None:
            ts_low = time_series[2*i]
            ts_up = time_series[2*i + 1]
            l = plt.scatter(timestamps, ts_low, marker=".", s=1 ,color=colors[color_idx])
            l = plt.scatter(timestamps, ts_up, marker=".", s=1 ,color=colors[color_idx])
            plt.fill_between(timestamps, ts_low, ts_up, facecolor="none", edgecolor=colors[i], hatch=hatches[color_idx])
            handle = patches.Patch(fill=None,edgecolor=colors[i], hatch=hatches[i])
            legend_handles.append(handle)
            #if class_labels is not None:
            #    l.set_label(class_labels[i])
            #else:
            #    l.set_label("Neuron " + str(color_idx))
    plt.hlines(y=0, xmin=0, xmax=len(timestamps), linestyles="dashed")

    if xticks is not None:
        ax.set_xticks(np.arange(0,len(xticks)))
        ax.set_xticklabels(xticks, rotation=45)
    plt.legend(legend_handles, class_labels)
    plt.title(title if title is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.xlabel(xlabel if xlabel is not None else "")
    plt.subplots_adjust(bottom=0.2)


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
