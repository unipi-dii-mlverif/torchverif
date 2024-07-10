from torchverif.interval_tensor.v2 import IntervalTensor
import torch
from scipy.stats import norm
from scipy import stats

class Simulator:
    def __init__(self, program, internal_samples=1000):
        torch.set_grad_enabled(False)
        self.program = program
        self.internal_samples = internal_samples
        self.simulation_samples = None
        self.simulation_iterations = None

    def simulate(self, input: IntervalTensor, iterations, label="", seed=9999):
        torch.manual_seed(seed)
        samples = input.sample(self.internal_samples * iterations)
        self.simulation_samples = self.program(samples)
        self.simulation_samples = torch.reshape(self.simulation_samples, (iterations, self.internal_samples, -1))
        self.simulation_iterations = iterations
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

    def cdf(self, index):
        index_samples = self.simulation_samples[:,:,index]
        flattened = torch.flatten(index_samples)
        normalized_cdf = stats.ecdf(flattened)
        return normalized_cdf

    def minmax_query(self, confidence=0.99):
        return self.query(torch.min, confidence), self.query(torch.max, confidence)

