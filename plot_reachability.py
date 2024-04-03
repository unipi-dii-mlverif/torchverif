from net_interval import *
import torch.nn as nn
if __name__ == '__main__':
    # Input regions
    f1 = [38, 46]
    f2 = [1, 3]
    f3 = [9, 10]
    f4 = [1, 5]

    net = nn.Sequential(
        nn.Linear(in_features=4, out_features=4, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=4, out_features=2, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=2, out_features=4, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=4, out_features=2, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=2, out_features=4, bias=True),
        nn.Sigmoid()
    )

    #net = torch.load("./models/attack_nn_4layers.pth", map_location=torch.device('cpu'))
    #net = torch.nn.Sequential(*(list(net.children())[:-1]))

    intervals, bounds = evaluate_net_interval(net, [f1, f2, f3, f4])
    o_sam = evaluate_net_samples(net, [f1, f2, f3, f4], cartesian=True, samples=10)
    interval_plot_scores_helper(o_sam, bounds, 0)
    print(verify_bound_disjunction(intervals,1))