import interval_tensor.v2
from net_interval import *


def test_mpc_seq_v2():
    # Input regions
    f1 = [24, 25]
    f2 = [0, 35]
    f3 = [-2, 2]
    net = torch.load("../models/model.pth", map_location=torch.device('cpu'))

    arr_f = torch.Tensor([f1, f2, f3])
    interval = interval_tensor.v2.IntervalTensor(arr_f[:, 0], arr_f[:, 1])
    o = net(interval)
    bounds = bounds_from_v2_predictions(o)

    interval_plot_scores_helper([], np.array(bounds), threshold=0,
                                class_labels=["Acceleration bounds"],
                                xlabel="Acceleration (m/s$^2$)",
                                ylabel="Output neuron", legend=1)
    plt.show()


def test_mpc_multiple_seq_v2():
    f1 = [24, 25]
    f3 = [-2, 2]

    bound_list = []
    ticks = []
    for i in range(0, 50, 4):
        f = [i, i + 5]
        ticks.append("[" + str(f[0]) + "," + str(f[1]) + "]")
        arr_f = torch.Tensor([f1, f, f3])
        interval = interval_tensor.v2.IntervalTensor(arr_f[:, 0], arr_f[:, 1])
        net = torch.load("../models/model.pth", map_location=torch.device('cpu'))
        bounds = bounds_from_v2_predictions(net(interval))
        bound_list.append(np.array(bounds))

    interval_time_plot_helper(bound_list, neuron=None,
                              class_labels=["Acceleration bounds"],
                              xticks=ticks,
                              xlabel="Ego-car relative distance uncertainty",
                              ylabel="Acceleration (m/s$^2$)")
    plt.show()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    test_mpc_seq_v2()
    test_mpc_multiple_seq_v2()
