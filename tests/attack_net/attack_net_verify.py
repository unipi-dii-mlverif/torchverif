from torchverif.net_interval.v2 import *
from torchverif.net_interval.v1 import evaluate_fcnn_samples, evaluate_fcnn_interval, verify_bound_disjunction
import matplotlib.pyplot as plt
import torch

def test_attack_nn():
    # Input regions
    f1 = [38, 46]  # MEAN VALUE DISTANCE EGO-LEAD
    f2 = [1, 3]  # STD DISTANCE EGO-LEAD
    f3 = [7, 13]  # MEAN VALUE RELATIVE SPEED EGO-LEAD
    f4 = [3, 6]  # STD RELATIVE SPEED EGO-LEAD
    f5 = [19, 21]  # MEAN VALUE EGO-SPEED
    f6 = [0.4, 0.7]  # STD VALUE EGO-SPEED

    arr_f = [f5, f6, f1, f2, f3, f4]

    net = torch.load("../../models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
    net = torch.nn.Sequential(*(list(net.children())[:-1]))

    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    print(bounds)
    o_sam = evaluate_fcnn_samples(net, arr_f, cartesian=False, samples=10000)
    interval_plot_scores_helper([], bounds, threshold=0,
                                class_labels=["no attack", "attack"],
                                xlabel="Prediction score",
                                ylabel="Class label", legend=1)
    print(verify_bound_disjunction(intervals, 1))
    plt.show()


def test_multiple_attack_nn():
    # Input regions
    f1 = [38, 46]  # MEAN VALUE DISTANCE EGO-LEAD
    f2 = [1, 3]  # STD DISTANCE EGO-LEAD
    f3 = [7, 13]  # MEAN VALUE RELATIVE SPEED EGO-LEAD
    f4 = [3, 6]  # STD RELATIVE SPEED EGO-LEAD
    f5 = [19, 21]  # MEAN VALUE EGO-SPEED
    f6 = [0.4, 0.7]  # STD VALUE EGO-SPEED

    bound_list = []
    ticks = []
    for i in range(1, 15, 1):
        f = [i, i + 3]
        ticks.append("[" + str(f[0]) + "," + str(f[1]) + "]")
        arr_f = [f5, f6, f1, f2, f, f4]

        net = torch.load("../../models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
        net = torch.nn.Sequential(*(list(net.children())[:-1]))

        intervals, bounds = evaluate_fcnn_interval(net, arr_f)
        bound_list.append(bounds)

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["no attack", "attack"],
                              xticks=ticks,
                              xlabel="Ego-car mean absolute speed uncertainty",
                              ylabel="Prediction score")
    plt.show()


if __name__ == '__main__':
    test_attack_nn()
    test_multiple_attack_nn()
