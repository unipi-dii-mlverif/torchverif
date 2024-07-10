from matplotlib import patches

import torchverif.interval_tensor.v2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def bounds_from_v2_predictions(predictions: torchverif.interval_tensor.v2.IntervalTensor):
    bounds = []
    for i, b in enumerate(predictions):
        bounds.append([i, b._inf.item()])
        bounds.append([i, b._sup.item()])
    return np.array(bounds)


def class_bounds_from_net_outputs(outputs, out_classes):
    outputs = outputs.detach().numpy()
    num_samples = outputs.shape[0]
    poly = []
    bounds = []
    for i in range(out_classes):
        poly_points = np.reshape(outputs[:, i], (num_samples, 1))
        min_p = np.min(poly_points)
        max_p = np.max(poly_points)
        bounds.append([i, min_p])
        bounds.append([i, max_p])
        poly_p_i = np.hstack((i * np.ones((len(poly_points), 1)), poly_points))
        poly.append(poly_p_i)
    return poly, np.array(bounds)


def interval_plot_scores_helper(sample_group, bounds, threshold=None, legend=None, class_labels=None, xticks=[],
                                xlabel=None, ylabel=None, title=None):
    colors = cm.rainbow(np.linspace(0, 1, len(bounds)))
    yt = []
    for i, sample in enumerate(sample_group):
        l = plt.scatter(sample[:, 1], sample[:, 0], s=0.5, marker=".", color=colors[i])
        yt.append(i)
        if legend is not None and legend > 1:
            l.set_label(class_labels[i])

    bbs = []
    for b in bounds:
        if len(bbs) < b[0] + 1:
            bbs.append([])
        bbs[int(b[0])].append(b[1])

    for i, b in enumerate(bbs):
        l = plt.scatter(y=[i, i], x=[bounds[2 * i, 1], bounds[2 * i + 1, 1]], marker="*", color=colors[i])
        if len(sample_group) < 1:
            l = plt.hlines(y=i, xmin=b[0], xmax=b[1], color=colors[i], linestyles="--")
        if legend is not None and legend >= 1:
            l.set_label(class_labels[i])

    if threshold is not None:
        plt.axvline(threshold, color="green", linestyle="--")
    plt.ylabel("Output neuron")
    plt.xlabel("Prediction")

    if len(bounds) > 0:
        plt.yticks(bounds[:, 0])
    else:
        plt.yticks(yt)
    if legend is not None and legend > 0:
        plt.legend()
    plt.title(title if title is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.xlabel(xlabel if xlabel is not None else "")


def interval_time_plot_helper(bound_list, neuron=None, class_labels=None, xticks=[], xlabel=None, ylabel=None,
                              title=None, threshold=None):
    hatches = ["O", "//", "xx", "\\\\", "//", "O", "//", "xx", "\\\\", "//"]
    class_number = int(len(bound_list[0]) / 2)
    colors = ["red"]
    time_series = np.empty([len(bound_list[0]), len(bound_list)])  # as the number of output labels*2 (up and inf)
    timestamps = np.linspace(0, len(bound_list), len(bound_list))
    fig, ax = plt.subplots(1, 1)

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
        if (neuron is not None and color_idx == neuron) or neuron is None:
            ts_low = time_series[2 * i]
            ts_up = time_series[2 * i + 1]
            l = plt.scatter(timestamps, ts_low, marker=".", s=1, color=colors[color_idx])
            l = plt.scatter(timestamps, ts_up, marker=".", s=1, color=colors[color_idx])
            plt.fill_between(timestamps, ts_low, ts_up, facecolor="none", edgecolor=colors[i], hatch=hatches[color_idx])
            handle = patches.Patch(fill=None, edgecolor=colors[i], hatch=hatches[i])
            legend_handles.append(handle)
            # if class_labels is not None:
            #    l.set_label(class_labels[i])
            # else:
            #    l.set_label("Neuron " + str(color_idx))
    if threshold is not None:
        plt.hlines(y=threshold, xmin=0, xmax=len(timestamps), linestyles="dashed")

    if xticks is not None:
        ax.set_xticks(np.arange(0, len(xticks)))
        ax.set_xticklabels(xticks, rotation=45)
    plt.legend(legend_handles, class_labels)
    plt.title(title if title is not None else "")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.xlabel(xlabel if xlabel is not None else "")
    plt.subplots_adjust(bottom=0.2)
