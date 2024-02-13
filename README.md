# torch2pvs
Torch MLP model to PVS theory converter

### Prerequisites

Install `torch` and `numpy` with:

```bash
pip install -r requirements.txt
```

### Usage

Output PVS buffer to standard output:

```bash
python torch2pvs.py examples/model.pth
```

Output PVS buffer to file:

```bash
python torch2pvs.py examples/model.pth -p outputs/model.pvs
```

