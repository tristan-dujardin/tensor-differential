import pytest
import torch
import numpy as np

from src.Tensor import Tensor
from src.functions import log_d, exp_d, sin_d, cos_d, tan_d, tanh_d, sqrt_d

# Define the functions
def f(x):
    return x**2 + 2*x + 1

def f7(x):
    return x**3 + 3*x**2 - 2*x + 5

def f8(x):
    return x**4 + 4*x**3 - 3*x**2 + 6*x + 7

def f9(x):
    return x**5 + 5*x**4 - 4*x**3 + 8*x**2 + 9*x + 10

def f10(x):
    return exp_d(x)

def f10_torch(x):
    return torch.exp(x)

def f11(x):
    return log_d(x)

def f11_torch(x):
    return torch.log(x)

def f12(x):
    return sin_d(x)

def f12_torch(x):
    return torch.sin(x)

def f13(x):
    return cos_d(x)

def f13_torch(x):
    return torch.cos(x)

def f14(x):
    return tan_d(x)

def f14_torch(x):
    return torch.tan(x)

def f17(x):
    return tanh_d(x)

def f17_torch(x):
    return torch.tanh(x)

def f18(x):
    return 1/x

def f19(x):
    return sqrt_d(x)

def f19_torch(x):
    return torch.sqrt(x)

def f22(x):
    return log_d(x)

def f22_torch(x):
    return torch.log(x)

def f27(x):
    return x**3 + 3*x**2 - 2*x + sin_d(x) + 5

def f27_torch(x):
    return x**3 + 3*x**2 - 2*x + torch.sin(x) + 5

def f28(x):
    return x**4 + 4*x**3 - 3*x**2 + 6*x + cos_d(x) + 7

def f28_torch(x):
    return x**4 + 4*x**3 - 3*x**2 + 6*x + torch.cos(x) + 7

def f29(x):
    return x**5 + 5*x**4 - 4*x**3 + 8*x**2 + 9*x + sin_d(x) + cos_d(x) + 10

def f29_torch(x):
    return x**5 + 5*x**4 - 4*x**3 + 8*x**2 + 9*x + torch.sin(x) + torch.cos(x) + 10

def f30(x):
    return sin_d(x)**2 + cos_d(x)**2 + x**2

def f30_torch(x):
    return torch.sin(x)**2 + torch.cos(x)**2 + x**2

def f31(x):
    return sin_d(x)**3 + cos_d(x)**3 + x**3

def f31_torch(x):
    return torch.sin(x)**3 + torch.cos(x)**3 + x**3

# List of function pairs
function_pairs = [
    (f, f),
    (f7, f7),
    (f8, f8),
    (f9, f9),
    (f10, f10_torch),
    (f11, f11_torch),
    (f12, f12_torch),
    (f13, f13_torch),
    (f14, f14_torch),
    (f17, f17_torch),
    (f18, f18),
    (f19, f19_torch),
    (f22, f22_torch),
    (f27, f27_torch),
    (f28, f28_torch),
    (f29, f29_torch),
    (f30, f30_torch),
    (f31, f31_torch),
]

@pytest.mark.parametrize("func, func_torch", function_pairs)
def test_function_gradients(func, func_torch):
    x = Tensor(0.1)
    x_torch = torch.tensor(x.data, requires_grad=True)

    y = func(x)
    y_torch = func_torch(x_torch)

    y.backward()
    y_torch.backward()

    assert np.isclose(x.grad, x_torch.grad.item(), atol=1e-6), \
        f"Gradients do not match for function {func.__name__}: {x.grad} vs {x_torch.grad.item()}"

if __name__ == '__main__':
    pytest.main()
