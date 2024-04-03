import matplotlib.pyplot as plt
import numpy as np
import torch
from interval_tensor import IntervalTensor


def extract_feature_tensor_bounds(feature_tensor):
    idata = feature_tensor.data()
    out_data = [np.array([])] * len(idata)
    for l_, i_ in enumerate(idata):
        feat_ = np.array(i_[0]).reshape(2, 1)
        label_ = (np.ones(2) * l_).reshape(2, 1)
        stacked = np.hstack((label_, feat_))
        out_data[l_] = stacked
    return out_data


# Take a random network
# Simple example, 2 feature in 1 feature out
# The internal layers do not matter for now, except for the activation functions
net = torch.load("./examples/attack_nn_4layers.pth", map_location=torch.device('cpu'))
print(net)

net = torch.nn.Sequential(*(list(net.children())[:-1]))
input_features = net[0].weight.shape[1]
input_size = net[0].weight.shape[0]
# Select the sampling region for the inputs and sampling interval
samples = 10000

# set the seed for reproducibility
torch.random.manual_seed(8086)

# Input regions
f1 = [38, 46]
f2 = [1, 3]
f3 = [9, 10]
f4 = [1, 5]

# Per feature uniform sampling
feat_1 = torch.distributions.uniform.Uniform(f1[0], f1[1]).sample([samples])
feat_2 = torch.distributions.uniform.Uniform(f2[0], f2[1]).sample([samples])
feat_3 = torch.distributions.uniform.Uniform(f3[0], f3[1]).sample([samples])
feat_4 = torch.distributions.uniform.Uniform(f4[0], f4[1]).sample([samples])

# Combine the features
feat = torch.stack([feat_1, feat_2, feat_3, feat_4], 1)
print(feat)
# Evaluate the network in the sampled points
sample_output = net(feat)
poly_points = sample_output.detach().numpy()

poly_points_class_0 = np.reshape(poly_points[:, 0], (samples, 1))
poly_points_class_1 = np.reshape(poly_points[:, 1], (samples, 1))

p0 = np.hstack((np.zeros((len(poly_points_class_0), 1)), poly_points_class_0))
p1 = np.hstack((np.ones((len(poly_points_class_1), 1)), poly_points_class_1))

intervals = np.array([f1, f2, f3, f4], dtype=object)
i = IntervalTensor(intervals)
o = net(i)
bounds = extract_feature_tensor_bounds(o)
bb = np.array(bounds).reshape(4, 2)

plt.scatter(p0[:, 1], p0[:, 0], marker=".")
plt.scatter(p1[:, 1], p1[:, 0], marker=".")
plt.scatter(bb[:, 1], bb[:, 0], marker="*")
plt.ylabel("Label")
plt.xlabel("Score")
plt.yticks(bb[:, 0])
plt.axvline(x=0, color="green", linestyle="--")
plt.savefig("reachability_attack_nn_undec.png")
plt.show()
