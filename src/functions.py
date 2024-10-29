from .Tensor import Tensor
import numpy as np

def log_d(dual_number: Tensor):
    out = Tensor(np.log(dual_number.data), (dual_number,), 'log')

    def _backward():
        dual_number.grad += (1 / dual_number.data) * out.grad
    out._backward = _backward

    return out

def exp_d(dual_number: Tensor):
    out = Tensor(np.exp(dual_number.data), (dual_number,), 'exp')

    def _backward():
        dual_number.grad += np.exp(dual_number.data) * out.grad
    out._backward = _backward

    return out

def sin_d(dual_number: Tensor):
    out = Tensor(np.sin(dual_number.data), (dual_number,), 'sin')

    def _backward():
        dual_number.grad += np.cos(dual_number.data) * out.grad
    out._backward = _backward

    return out

def cos_d(dual_number: Tensor):
    out = Tensor(np.cos(dual_number.data), (dual_number,), 'cos')

    def _backward():
        dual_number.grad += - np.sin(dual_number.data) * out.grad
    out._backward = _backward

    return out

def sigmoid_d(dual_number: Tensor):
    sig_value = 1 / (1 + np.exp(- dual_number.data))
    out = Tensor(sig_value, (dual_number,), 'sigmoid')

    def _backward():
        dual_number.grad += sig_value * (1 - sig_value) * out.grad
    out._backward = _backward

    return out

def tanh_d(dual_number: Tensor):
    tanh_value = np.tanh(dual_number.data)
    out = Tensor(tanh_value, (dual_number,), 'tanh')

    def _backward():
        dual_number.grad += (1 - tanh_value ** 2) * out.grad
    out._backward = _backward

    return out

def tan_d(dual_number: Tensor):
    out = Tensor(np.tan(dual_number.data), (dual_number,), 'tan')

    def _backward():
        dual_number.grad += (1 / np.cos(dual_number.data) ** 2) * out.grad
    out._backward = _backward

    return out

def sqrt_d(dual_number: Tensor):
    out = Tensor(np.sqrt(dual_number.data), (dual_number,), 'sqrt')

    def _backward():
        dual_number.grad += (0.5 / np.sqrt(dual_number.data)) * out.grad
    out._backward = _backward

    return out

def pow_d(dual_number: Tensor, power: int):
    out = Tensor(dual_number.data ** power, (dual_number,), f'pow {power}')

    def _backward():
        dual_number.grad += power * (dual_number.data ** (power - 1)) * out.grad
    out._backward = _backward

    return out

def softmax_d(dual_number: Tensor):
    out = Tensor(1.0, (dual_number,), 'softmax')

    def _backward():
        dual_number.grad += 0.0
    out._backward = _backward

    return out
