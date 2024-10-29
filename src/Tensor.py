class Tensor:
    
    """ stores a single scalar Tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):

        self.data = data
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        out._prev = set([self, other])
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        out._prev = set([self, other])
        return out

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data ** other, (self,), f'** {other}')

            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward

            out._prev = set([self])
            return out
        else:
            raise TypeError("Power must be an integer or a float.")

    def relu(self):
        out = Tensor(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        out._prev = set([self])
        return out

    def backward(self):
        order = []
        visited = set()

        def build_order(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_order(child)
                order.append(tensor)
        
        build_order(self)

        self.grad = 1.0
        for tensor in reversed(order):
            tensor._backward()

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
        return f"Tensor(data={self.data}, grad={self.grad})"