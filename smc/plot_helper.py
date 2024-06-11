import numpy as np
import torch
from matplotlib import pyplot as plt, cm


def format_query_bounds(min_bounds: torch.Tensor, max_bounds: torch.Tensor):
    min_bounds = min_bounds.detach().numpy()
    max_bounds = max_bounds.detach().numpy()

    out_classes = min_bounds.shape[0]
    bounds = []
    for i in range(out_classes):
        bounds.append([int(i), min_bounds[0, i]])
        bounds.append([int(i), max_bounds[1, i]])
    return np.array(bounds)


def format_query_output(output: torch.Tensor):
    output = output.detach().numpy()
    out_classes = output.shape[0]
    bounds = []
    for i in range(out_classes):
        bounds.append([int(i), output[0, i]])
        bounds.append([int(i), output[1, i]])
    return np.array(bounds)


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


def show_plot():
    plt.show()


def save_plot(filename):
    plt.savefig(filename)
