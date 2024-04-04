import torch

from net_interval import *
import torch.nn as nn
import matplotlib.pyplot as plt


def test_seq():
    # Input regions
    f1 = [38, 46]
    f2 = [1, 3]
    f3 = [9, 10]
    f4 = [1, 5]

    arr_f = [f1, f2, f3, f4, f1, f3]

    net = torch.load("./models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
    net = torch.nn.Sequential(*(list(net.children())[:-1]))

    intervals, bounds = evaluate_fcnn_interval(net, arr_f)
    o_sam = evaluate_fcnn_samples(net, arr_f, cartesian=False, samples=10000)
    interval_plot_scores_helper([], bounds, threshold=0)
    print(verify_bound_disjunction(intervals, 0))


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
    test_seq()
