from interval_tensor.v2 import IntervalTensor
import torch
from scipy.stats import norm
import plot_helper
class Simulator:
    def __init__(self, program):
        torch.set_grad_enabled(False)
        self.program = program
        self.internal_samples = 1000
        self.simulation_samples = None
        self.simulation_iterations = None

    def simulate(self, input: IntervalTensor, iterations, label="", seed=9999):
        torch.manual_seed(seed)
        samples = input.sample(self.internal_samples * iterations)
        self.simulation_samples = self.program(samples)
        self.simulation_samples = torch.reshape(self.simulation_samples, (iterations, self.internal_samples, -1))
        self.simulation_iterations = iterations
        print(self.simulation_samples.shape)
        return self.simulation_samples

    def query(self, functor, confidence=0.99):
        if functor == torch.max or functor == torch.min:
            query_result,_  = functor(self.simulation_samples, dim=1, keepdim=True)
        else:
            query_result = functor(self.simulation_samples, dim=1, keepdim=True)

        query_predictor = torch.mean(query_result, dim=0, keepdim=True)
        query_stddev = torch.std(query_result, dim=0, keepdim=True)
        zscore = norm.ppf(confidence)
        conf = query_stddev * zscore
        query_bounds = torch.hstack([query_predictor - conf, query_predictor + conf])
        query_bounds = torch.squeeze(query_bounds, 0)
        return query_bounds


if __name__ == '__main__':
    net = torch.load("../models/attack_nn_4layers_6feat.pth", map_location=torch.device('cpu'))
    net = torch.nn.Sequential(*(list(net.children())[:-1]))

    simulator = Simulator(net)
    f1 = [38, 46]  # MEAN VALUE DISTANCE EGO-LEAD
    f2 = [1, 3]  # STD DISTANCE EGO-LEAD
    f3 = [7, 13]  # MEAN VALUE RELATIVE SPEED EGO-LEAD
    f4 = [3, 6]  # STD RELATIVE SPEED EGO-LEAD
    f5 = [19, 21]  # MEAN VALUE EGO-SPEED
    f6 = [0.4, 0.7]  # STD VALUE EGO-SPEED

    arr_f = torch.Tensor([f5, f6, f1, f2, f3, f4])
    interval = IntervalTensor(arr_f[:, 0], arr_f[:, 1])
    simulator.simulate(interval, 100)
    a = simulator.query(torch.max, confidence=0.9999)
    b = simulator.query(torch.min, confidence=0.9999)
    c = simulator.query(torch.mean, confidence=0.9999)

    print(c)

    bo = plot_helper.format_query_bounds(a,b)
    plot_helper.interval_plot_scores_helper([], bo, threshold=0)
    plot_helper.show_plot()
