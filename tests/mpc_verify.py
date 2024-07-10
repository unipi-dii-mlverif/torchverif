from torchverif.net_interval.v1 import *
from torchverif.pvs import *
import matplotlib.pyplot as plt


def test_pvs_emit():
    emit_pvs_from_pth("../models/model.pth")

def test_mpc_seq():
    # Input regions
    f1 = [24, 25]
    f2 = [0, 35]
    f3 = [-2, 2]
    arr_f = [f1, f2, f3]
    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    print(bounds)
    interval_plot_scores_helper([], bounds, threshold=0,
                                class_labels=["Acceleration bounds"],
                                xlabel="Acceleration (m/s$^2$)",
                                ylabel="Output neuron", legend=1)
    print(bounds)
    plt.show()


def test_mpc_multiple_seq():
    f1 = [24, 25]
    f3 = [-2, 2]

    bound_list = []
    ticks = []
    for i in range(0, 50, 4):
        f = [i, i + 5]
        ticks.append("[" + str(f[0]) + "," + str(f[1]) + "]")
        arr_f = [f1, f, f3]

        net = torch.load("../models/model.pth", map_location=torch.device('cpu'))

        intervals, bounds = evaluate_fcnn_interval(net, arr_f)
        bound_list.append(bounds)

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["Acceleration bounds"],
                              xticks=ticks,
                              xlabel="Ego-car relative distance uncertainty",
                              ylabel="Acceleration (m/s$^2$)")
    plt.show()


if __name__ == '__main__':
    test_mpc_seq()
    test_mpc_multiple_seq()
    test_pvs_emit()
