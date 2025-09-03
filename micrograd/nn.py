from engine import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        assert len(x) == len(self.w), f"Input size {len(x)} doesn't match weight size {len(self.w)}"
        
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self): 
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs): 
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):  
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self): 
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nout, hidden_layers, **kwargs):  
        self.layers = []
        last_size = nin
        for h in hidden_layers:
            self.layers.append(Layer(last_size, h, **kwargs))
            last_size = h
        self.layers.append(Layer(last_size, nout, **kwargs))

    def __call__(self, x):  
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):  
        return [p for layer in self.layers for p in layer.parameters()]
        
    def __repr__(self):  
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

if __name__ == "__main__":
    model = MLP(2, 1, [4, 4])
    print(model)
    
    # MLP of [Layer of [ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2)], 
    #         Layer of [ReLUNeuron(4), ReLUNeuron(4), ReLUNeuron(4), ReLUNeuron(4)], 
    #         Layer of [ReLUNeuron(4)]]
