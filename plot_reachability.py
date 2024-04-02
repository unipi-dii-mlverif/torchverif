import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import interval_tensor
from interval_tensor import IntervalTensor


def extract_output_tensor(interval_tensor):
    idata = interval_tensor.data()
    out_data = [0] * len(idata)
    for l_, i_ in enumerate(idata):
        feat_ = np.array(i_[0]).reshape(2, 1)
        label_ = (np.ones(2) * l_).reshape(2, 1)
        stacked = np.hstack((label_, feat_))
        out_data[l_] = stacked
    return out_data


# Take a random network
# Simple example, 2 feature in 1 feature out
# The internal layers do not matter for now, except for the activation functions
net = nn.Sequential(
    nn.Linear(in_features=2, out_features=10, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=4, bias=True),
    nn.Sigmoid()
)

input_features = net[0].weight.shape[1]

input_size = net[0].weight.shape[0]

input_features

# Select the sampling region for the inputs and sampling interval
samples = 10000

# set the seed for reproducibility
torch.random.manual_seed(100)

# Per feature uniform sampling
feat_1 = torch.distributions.uniform.Uniform(-1, 1).sample([samples])
feat_2 = torch.distributions.uniform.Uniform(-1, 1).sample([samples])

# Combine the features
feat = torch.stack([feat_1, feat_2], 1)

# Evaluate the network in the sampled points
sample_output = net(feat)
poly_points = sample_output.detach().numpy()

poly_points_class_0 = np.reshape(poly_points[:, 0], (samples, 1))
poly_points_class_1 = np.reshape(poly_points[:, 1], (samples, 1))
poly_points_class_2 = np.reshape(poly_points[:, 2], (samples, 1))
poly_points_class_3 = np.reshape(poly_points[:, 3], (samples, 1))

p0 = np.hstack((np.zeros((len(poly_points_class_0), 1)), poly_points_class_0))
p1 = np.hstack((np.ones((len(poly_points_class_1), 1)), poly_points_class_1))
p2 = np.hstack((2*np.ones((len(poly_points_class_2), 1)), poly_points_class_2))
p3 = np.hstack((3*np.ones((len(poly_points_class_3), 1)), poly_points_class_3))

intervals = np.array([[-1, 1], [-1, 1]], dtype=object)
i = IntervalTensor(intervals)
o = net(i)

bounds = extract_output_tensor(o)
bb = np.array(bounds).reshape(8, 2)

plt.scatter(p0[:, 1], p0[:, 0], s=10, marker="+")
plt.scatter(p1[:, 1], p1[:, 0], s=10, marker="+")
plt.scatter(p2[:, 1], p2[:, 0], s=10, marker="+")
plt.scatter(p3[:, 1], p3[:, 0], s=10, marker="+")
plt.scatter(bb[:, 1], bb[:, 0], color="green", marker="*")
plt.ylabel("Label")
plt.xlabel("Score")
plt.yticks(bb[:, 0])
plt.show()
