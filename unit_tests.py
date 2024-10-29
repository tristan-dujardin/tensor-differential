import torch
from src.Tensor import Tensor


def test_tensor_affine():
    x = Tensor(1.0)
    y = Tensor(2.0)
    out = x * 2.0 + y
    out.backward()

    x_torch = torch.tensor(1.0, requires_grad=True)
    y_torch = torch.tensor(2.0, requires_grad=True)
    out_torch = x_torch * 2.0 + y_torch
    out_torch.backward()

    assert torch.allclose(x_torch.grad, torch.tensor(2.0), atol=1e-6), \
        f"Expected x.grad to be 2.0, but got {x_torch.grad.item()}"
    assert torch.allclose(y_torch.grad, torch.tensor(1.0), atol=1e-6), \
        f"Expected y.grad to be 1.0, but got {y_torch.grad.item()}"

def test_tensor_neg():
    x = Tensor(1.0)
    out = -x
    out.backward()

    x_torch = torch.tensor(1.0, requires_grad=True)
    out_torch = -x_torch
    out_torch.backward()

    assert torch.allclose(x_torch.grad, torch.tensor(-1.0), atol=1e-6), \
        f"Expected x.grad to be -1.0, but got {x_torch.grad.item()}" 
    
def test_tensor_div():
    x = Tensor(1.0)
    y = Tensor(2.0)
    out = x / y
    out.backward()

    x_torch = torch.tensor(1.0, requires_grad=True)
    y_torch = torch.tensor(2.0, requires_grad=True)
    out_torch = x_torch / y_torch
    out_torch.backward()

    assert torch.allclose(x_torch.grad, torch.tensor(0.5), atol=1e-6), \
        f"Expected x.grad to be 0.5, but got {x_torch.grad.item()}"
    assert torch.allclose(y_torch.grad, torch.tensor(-0.25), atol=1e-6), \
        f"Expected y.grad to be -0.25, but got {y_torch.grad.item()}"

if __name__ == '__main__':
    test_tensor_affine()
    test_tensor_neg()
    test_tensor_div()