# microgradproject

Tiny autograd engine and a minimal neural network library (pure Python), inspired by micrograd. It implements a scalar Value type with reverse‑mode automatic differentiation and a small MLP stack with Neuron/Layer abstractions.

## Features
- Value class with gradients and a proper backward pass (topological order)
- Supports +, -, *, /, power, ReLU, and broadcasting with Python scalars
- Simple neural building blocks: Neuron, Layer, MLP
- Zero external dependencies (Python standard library only)

## Project structure
```
.
├─ engine.py   # Autograd Value and ops + demo
├─ nn.py       # Module/Neuron/Layer/MLP + summary demo
└─ __init__.py # Package marker
```

## Quick start
Requirements: Python 3.8+ (no extra packages needed)

Run the built‑in demos:
- Autograd demo
  - Shows a small graph, calls backward, and prints gradients.
- MLP demo
  - Prints a model summary for a small network.

### Autograd demo
```powershell
py engine.py
```
Expected output:
```
Value(data=2.0, grad=42.0)
Value(data=3.0, grad=28.0)
Value(data=6.0, grad=14.0)
Value(data=7.0, grad=14.0)
Value(data=49.0, grad=1.0)
```

### MLP summary
```powershell
py nn.py
```
Example output:
```
MLP of [Layer of [ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2)], Layer of [ReLUNeuron(4), ReLUNeuron(4), ReLUNeuron(4), ReLUNeuron(4)], Layer of [ReLUNeuron(4)]]
```

## Usage examples
### Autograd in code
```python
from engine import Value

x1 = Value(2.0)
x2 = Value(-3.0)
x3 = Value(10.0)
b = Value(6.8813735870195432)

# simple expression
n = x1 * x2 + x3
m = n + b
m.backward()

print(n, m)     # forward values
print(x1.grad, x2.grad, x3.grad)  # gradients
```

### Simple forward pass with MLP
```python
from nn import MLP
from engine import Value

# 2 inputs -> two hidden layers [4, 4] -> 1 output
mlp = MLP(2, 1, [4, 4])

# One sample forward pass
x = [Value(1.0), Value(-2.0)]
yhat = mlp(x)   # returns a Value when nout == 1
print(yhat)
```


