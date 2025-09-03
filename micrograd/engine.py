class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other ,self.__class__) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other ,self.__class__) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        other = other if isinstance(other ,(int, float)) else Value(other)
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        # Topological order all of the children in the graph
        topologically_sorted_nodes = []
        visited_nodes = set()
        
        def build_topological_order(current_node):
            if current_node not in visited_nodes:
                visited_nodes.add(current_node)
                for predecessor in current_node._prev:
                    build_topological_order(predecessor)
                topologically_sorted_nodes.append(current_node)
        
        build_topological_order(self)
        
        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topologically_sorted_nodes):
            node._backward()




    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


if __name__== "__main__":

    a = Value(2.0)
    b = Value(3.0)
    c = a * b       # Value(data=6.0, grad=0)
    d = c + 1       # Value(data=7.0, grad=0)
    e = d * d       # Value(data=49.0, grad=0)

    e.backward()    

    print(a)        # Value(data=2.0, grad=42.0)
    print(b)        # Value(data=3.0, grad=28.0)
    print(c)        # Value(data=6.0, grad=14.0)
    print(d)        # Value(data=7.0, grad=14.0)
    print(e)        # Value(data=49.0, grad=1.0)