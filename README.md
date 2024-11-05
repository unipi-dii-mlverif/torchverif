# torchverif
Multiple pytorch tools towards formal verification of neural networks

### Tools

- PyTorch to PVS theory generation
- Interval propagation through PyTorch network
- Statistical Model Checking (SMC) neural network verification

### Prerequisites (if needed)

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Install 
```bash
pip install torchverif
```

### Usage

See `tests/` for examples



### Cite this

If you use this repository in your works, please cite the following papers:

```
@article{rossi2024closedloop,
title = {Neural networks in closed-loop systems: Verification using interval arithmetic and formal prover},
journal = {Engineering Applications of Artificial Intelligence},
volume = {137},
pages = {109238},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.109238},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624013964},
author = {Federico Rossi and Cinzia Bernardeschi and Marco Cococcioni},
keywords = {Cyber–physical systems, Neural networks, Closed-loop control systems, Formal verification, Interval arithmetic},
}
```



# TorchVerif API Documentation

## `torchverif/interval_tensor/v2/__init__.py`

### Classes

- **`IntervalTensor`**: Represents tensors where each element holds an interval instead of a single value, used for interval arithmetic. The `IntervalTensor` class defines operations on interval tensors, supporting various tensor operations directly on intervals.

### Functions

- **Tensor Operations**: Functions like `Linear`, `Pad`, `Conv2d`, `MaxPool2D`, `BatchNorm2D`, `Cat`, `Sum`, `Squeeze`, `Unsqueeze`, `Flatten`, `RepeatInterleave`, `Max`, `Min`, `Matmul`, and `Var` provide interval-based versions of these tensor operations, essential for neural network layers.
- **Activation Functions**: `ReLU`, `Sigmoid`, `Tanh`, `Sqrt`, and `Square` implement activation functions designed to handle intervals.
- **Arithmetic Functions**: Functions such as `Add`, `Sub`, `Mean`, `Max`, `Min`, and `Var` support interval arithmetic across tensors.
- **Utility Methods**: Functions like `from_np_supinf`, `interval_from_infsup`, `from_raw`, and `check_interval` assist with constructing and validating intervals. Operators like `__add__`, `__sub__`, `__mul__`, and `__truediv__` are overloaded to support interval-based arithmetic directly on `IntervalTensor` objects.

---

## `torchverif/net_interval/v2/__init__.py`

The `net_interval` module provides interval analysis methods for neural network verification.

### Functions

- **`bounds_from_v2_predictions`**: Computes the interval bounds of the output given the network predictions. This function is used to verify that the network’s predictions stay within certain bounds.
- **`class_bounds_from_net_outputs`**: Extracts the bounds of each class’s output from network predictions, aiding in robustness verification by determining if outputs stay within secure intervals.
- **`interval_plot_scores_helper`**: Provides interval-based scoring for plotting, useful for visualizing bounds across classes or network layers.
- **`interval_time_plot_helper`**: Assists with plotting intervals over time, observing how bounds evolve during network evaluation or training.

---

## `torchverif/pvs/torch2pvs.py`

This module provides functionality for converting Torch models into PVS-compatible format, enabling formal verification processes.

### Functions

- **`get_leaky_relu_string`**: Generates a string representation for the Leaky ReLU activation function in PVS syntax.
- **`gen_vector_matrix_product`**: Generates PVS code for vector-matrix products, a common operation in neural network computations.
- **`gen_network_operation_sequence`**: Produces a sequence of network operations in PVS format, encoding the layers and structure of a Torch model.
- **`gen_theorem`** and **`gen_theorem_eval`**: Generate theorems for PVS verification, defining formal properties for the network to satisfy, and evaluate these theorems.
- **`gen_constraint_expressions`**: Produces PVS expressions that represent constraints, such as bounds or inequalities, for use in theorem proving.
- **`emit_pvs_from_pth`**: Converts a PyTorch model (`.pth` file) to PVS format, enabling formal verification of trained models in PVS.

---

## `torchverif/smc/plot_helper.py`

The `plot_helper` module provides functions to visualize interval analysis and verification results.

### Functions

- **`format_query_bounds`**: Prepares bounds for queries in a readable format, suitable for visualization and interpretation in plots.
- **`format_query_output`**: Structures output data for display and annotations in plots.
- **`sure_class`**: Identifies the most certain class based on interval bounds, useful for classification tasks where bounds indicate confidence levels.
- **`plot_cdf`**: Plots the cumulative distribution function (CDF) of a given interval, a valuable visualization tool for probabilistic or distributional data in interval analysis.
- **`interval_plot_scores_helper`**: Provides interval-based scoring for visual plots, highlighting ranges across different network outputs.
- **`finalize_plot`**: Finalizes plot aesthetics (axes, labels, legends) before rendering, ensuring clarity for interval or CDF visualizations.
- **`show_plot`**: Renders the plot, displaying interval or CDF results.
- **`save_plot`**: Saves the plot to a file, preserving interval analysis visualizations for reporting or documentation.

---

## `torchverif/smc/__init__.py`

This module contains tools for simulation-based verification methods.

### Classes

- **`Simulator`**: The main class for performing simulations and querying statistical bounds, allowing verification of probabilistic or interval-based properties.

### Functions (in `Simulator`)

- **`__init__`**: Initializes the simulator with configurations for simulations and statistical model checking.
- **`simulate`**: Runs a simulation, often involving interval-based testing or statistical verification.
- **`query`**: Executes a query on the simulation, retrieving bounds or probability estimates as needed.
- **`cdf`**: Calculates the cumulative distribution function (CDF) for simulation results, supporting probabilistic analysis.
- **`minmax_query`**: Queries minimum and maximum bounds in a simulation, useful for interval-based robustness verification.

---

This documentation provides a thorough overview of classes and functions within the `torchverif` library, focusing on interval analysis, formal verification, and visualization tools. Let me know if further detail on specific functions or modules is needed!
