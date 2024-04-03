import torch

from net_interval import *
import torch.nn as nn
import matplotlib.pyplot as plt


def test_seq():
    # Input regions
    f1 = [1.1, 2]
    f2 = [1.1, 2]
    f3 = [1.1, 2]
    f4 = [1.1, 2]
    torch.manual_seed(9999)

    net = nn.Sequential(
        nn.Linear(in_features=4, out_features=5, bias=True),
        nn.Sigmoid(),
        nn.Linear(in_features=5, out_features=4, bias=True)
    )

    # net = torch.load("./models/attack_nn_4layers.pth", map_location=torch.device('cpu'))
    # net = torch.nn.Sequential(*(list(net.children())[:-1]))

    intervals, bounds = evaluate_fcnn_interval(net, [f1, f2, f3, f4])
    o_sam = evaluate_fcnn_samples(net, [f1, f2, f3, f4], cartesian=False, samples=500)
    interval_plot_scores_helper(o_sam, bounds, 0)
    print(verify_bound_disjunction(intervals, 0))

def disp_bound_images(img_interval):
    inf_image = torch.Tensor(img_interval.inf())
    sup_image = torch.Tensor(img_interval.sup())
    f,a = plt.subplots(2)
    a[0].imshow(inf_image.permute(1,2,0))
    a[0].set_title('inf')
    a[1].imshow(sup_image.permute(1,2,0))
    a[1].set_title('sup')
    plt.show()

if __name__ == '__main__':
    # test_seq()
    r = [0.5, 0.8]
    sample = []
    for i in range(3*4*4):
        sample.append(r)
    np_sample = np.array(sample).reshape(3,4,4,2)
    it = IntervalTensor(np_sample)

    disp_bound_images(it)
