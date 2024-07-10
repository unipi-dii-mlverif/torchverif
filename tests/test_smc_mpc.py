from torchverif import smc
from torchverif.smc import plot_helper
from torchverif.interval_tensor.v2 import IntervalTensor
import torch
from scipy.stats import norm
from scipy import stats

if __name__ == '__main__':
    net = torch.load("../models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
    net = torch.nn.Sequential(*(list(net.children())[:-1]))

    simulator = smc.Simulator(net)
    f1 = [38, 46]  # MEAN VALUE DISTANCE EGO-LEAD
    f2 = [1, 3]  # STD DISTANCE EGO-LEAD
    f3 = [7, 13]  # MEAN VALUE RELATIVE SPEED EGO-LEAD
    f4 = [3, 6]  # STD RELATIVE SPEED EGO-LEAD
    f5 = [19, 21]  # MEAN VALUE EGO-SPEED
    f6 = [0.4, 0.7]  # STD VALUE EGO-SPEED

    arr_f = torch.Tensor([f5, f6, f1, f2, f3, f4])
    interval = IntervalTensor(arr_f[:, 0], arr_f[:, 1])
    simulator.simulate(interval, 100)
    cdf0 = simulator.cdf(0)
    cdf1 = simulator.cdf(1)

    plot_helper.plot_cdf(cdf0, legend="Class 0")
    plot_helper.plot_cdf(cdf1, legend="Class 1")
    plot_helper.show_plot(legend=True, xlabel="Prediction score")

    a = simulator.query(torch.max, confidence=0.9999)
    b = simulator.query(torch.min, confidence=0.9999)
    c = simulator.query(torch.mean, confidence=0.9999)

    print(a)
    print(b)
    print(c)
    bo = smc.plot_helper.format_query_bounds(a,b)
    smc.plot_helper.interval_plot_scores_helper([], bo)
    smc.plot_helper.show_plot()