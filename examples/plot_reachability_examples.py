import torch

import interval_tensor.v2
from net_interval import *
import torch.nn as nn
import matplotlib.pyplot as plt


def test_seq():
    # Input regions
    f1 = [38, 46]  # MEAN VALUE DISTANCE EGO-LEAD
    f2 = [1, 3]  # STD DISTANCE EGO-LEAD
    f3 = [7, 13]  # MEAN VALUE RELATIVE SPEED EGO-LEAD
    f4 = [3, 6]  # STD RELATIVE SPEED EGO-LEAD
    f5 = [19, 21]  # MEAN VALUE EGO-SPEED
    f6 = [0.4, 0.7]  # STD VALUE EGO-SPEED

    arr_f = [f5, f6, f1, f2, f3, f4]

    net = torch.load("../models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
    net = torch.nn.Sequential(*(list(net.children())[:-1]))

    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    print(bounds)
    o_sam = evaluate_fcnn_samples(net, arr_f, cartesian=False, samples=10000)
    interval_plot_scores_helper([], bounds, threshold=0,
                                class_labels=["no attack", "attack"],
                                xlabel="Prediction score",
                                ylabel="Class label", legend=1)
    print(verify_bound_disjunction(intervals, 1))
    # plt.savefig("plots/unsafe_detection.png")
    plt.show()


def multiple_seq():
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

        net = torch.load("./models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
        net = torch.nn.Sequential(*(list(net.children())[:-1]))

        intervals, bounds = evaluate_fcnn_interval(net, arr_f)
        bound_list.append(bounds)

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["no attack", "attack"],
                              xticks=ticks,
                              xlabel="Ego-car mean absolute speed uncertainty",
                              ylabel="Prediction score")
    plt.savefig("./plots/meanrelspeed_uncertainty.png")
    plt.show()


def test_ron_seq():
    # Input regions
    f1 = [5, 10]
    f2 = [0.4, 1]
    f3 = [1, 2]
    f4 = [4, 5]
    f5 = [9, 10]
    f6 = [3, 4]
    f7 = [1, 1.5]
    f8 = [0.8, 1]

    arr_f = [f1, f2, f3, f4, f5, f6, f7, f8]

    net = torch.load("./models/ron_net_50_10_5.pth", map_location=torch.device('cpu'))

    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    o_sam = evaluate_fcnn_samples(net, arr_f, cartesian=False, samples=10000)
    interval_plot_scores_helper([], bounds)


def test_mpc_seq():
    # Input regions
    f1 = [24, 25]
    f2 = [0, 35]
    f3 = [-2, 2]

    arr_f = [f1, f2, f3]

    net = torch.load("../models/model.pth", map_location=torch.device('cpu'))

    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    print(bounds)
    #o_sam = evaluate_fcnn_samples(net, arr_f, cartesian=False, samples=10000)
    interval_plot_scores_helper([], bounds, threshold=0,
                                class_labels=["Acceleration bounds"],
                                xlabel="Acceleration (m/s$^2$)",
                                ylabel="Output neuron", legend=1)
    print(bounds)
    plt.show()

def test_mpc_seq_v2():
    # Input regions
    f1 = [24, 25]
    f2 = [0, 35]
    f3 = [-2, 2]
    net = torch.load("../models/model.pth", map_location=torch.device('cpu'))

    arr_f = torch.Tensor([f1, f2, f3])
    interval = interval_tensor.v2.IntervalTensor(arr_f[:,0], arr_f[:,1])
    o = net(interval)
    bounds = bounds_from_v2_predictions(o)

    interval_plot_scores_helper([], np.array(bounds), threshold=0,
                                class_labels=["Acceleration bounds"],
                                xlabel="Acceleration (m/s$^2$)",
                                ylabel="Output neuron", legend=1)
    plt.show()


def mpc_multiple_seq():
    # Input regions
    f1 = [24, 25]
    f2 = [0, 34]
    f3 = [-2, 2]

    bound_list = []
    ticks = []
    for i in range(0, 50, 4):
        f = [i, i + 5]
        ticks.append("[" + str(f[0]) + "," + str(f[1]) + "]")
        arr_f = [f1, f, f3]

        net = torch.load("../models/cruise_model_hyp.pth", map_location=torch.device('cpu'))

        intervals, bounds = evaluate_fcnn_interval(net, arr_f)
        bound_list.append(bounds)

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["Acceleration bounds"],
                              xticks=ticks,
                              xlabel="Ego-car relative distance uncertainty",
                              ylabel="Acceleration (m/s$^2$)")
    plt.savefig("./plots/mpc_acceleration_vs_rdist_unc.png")
    plt.show()


def basic_net():
    net = nn.Sequential(
        nn.Linear(3, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )


    for i, param in enumerate(net.parameters()):
        print(param.data)
        if i == 0:
            param.data = nn.parameter.Parameter(torch.FloatTensor([[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, -1, -1]]))
        if i == 1:
            param.data = nn.parameter.Parameter(torch.FloatTensor([[1, 1, 1, 1], [-1, -1, -1, -1]]))
    torch.save(net, "models/save_basic.pth")

    f1 = [-1, 1]
    f2 = [-1, 1]
    f3 = [-1, 1]

    intervals, bounds = evaluate_fcnn_interval(net, [f1, f2, f3])
    interval_plot_scores_helper([], bounds,
                                class_labels=["Class 0", "Class 1"],
                                xlabel="Prediction score",
                                ylabel="Class label", legend=1, threshold=0)
    plt.savefig("./plots/basic_example.png")
    plt.show()


def disp_bound_images(img_interval):
    inf_image = torch.Tensor(img_interval.inf())
    sup_image = torch.Tensor(img_interval.sup())
    f, a = plt.subplots(2)
    a[0].imshow(inf_image.permute(1, 2, 0))
    a[0].set_title('inf')
    a[1].imshow(sup_image.permute(1, 2, 0))
    a[1].set_title('sup')
    plt.show()


if __name__ == '__main__':
    test_mpc_seq()
    test_mpc_seq_v2()
