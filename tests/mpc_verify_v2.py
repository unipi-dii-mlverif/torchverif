from torchverif.interval_tensor.v2 import IntervalTensor

from torchverif.net_interval.v2 import *
import torch
import numpy as np
import time

def test_mpc_seq_v2():

    # Input regions
    f1 = [24, 25]
    f2 = [0, 35]
    f3 = [-2, 2]

    start = time.time()
    net = torch.load("../models/cruise_model_hyp.pth", map_location=torch.device('cpu'))

    arr_f = torch.Tensor([f1, f2, f3])
    interval = IntervalTensor(arr_f[:, 0], arr_f[:, 1])
    o = net(interval)
    bounds = bounds_from_v2_predictions(o)

    interval_plot_scores_helper([], np.array(bounds), threshold=0,
                                class_labels=["Acceleration"],
                                xlabel="Acceleration",
                                ylabel="Command", legend=1)
    plt.show()
    
def test_mpc_multiple_seq_v2():
    f2 = [25.5, 25.5]
    f3 = [23.5, 23.5]

    bound_list = []
    ticks = []
    for i in range(18, 35, 1):
        f = [i, i + 0.5]
        ticks.append("[" + str(f[0]) + "," + str(f[1]) + "]")
        arr_f = torch.Tensor([f, f2, f3])
        interval = IntervalTensor(arr_f[:, 0], arr_f[:, 1])
        net = torch.load("../models/incubator.pth", map_location=torch.device('cpu'))
        bounds = bounds_from_v2_predictions(net(interval))
        bound_list.append(np.array(bounds))

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["Heat-on bounds"],
                              xticks=ticks,
                              xlabel="Box temperature",
                              ylabel="Heat-on score",
                              threshold=0.5)

    plt.show()


if __name__ == '__main__':
    test_mpc_seq_v2()
    test_mpc_multiple_seq_v2()