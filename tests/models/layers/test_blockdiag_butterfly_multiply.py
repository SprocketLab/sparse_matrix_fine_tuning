import sys

import pytest
import torch

sys.path.insert(0, "/fly")  # docker working dir
from src.ops.blockdiag_butterfly_multiply import (
    blockdiag_butterfly_multiply,
    blockdiag_butterfly_multiply_reference,
)


# the basic case for monarch
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("log_n", [4, 10, 12])
def test_block_diag_butterfly_multiply_reference(log_n, device, dtype):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    batch_size = 3
    x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
    w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    out1 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=1)
    out2 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2)
    out3 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=3)
    assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-4)
    assert torch.allclose(out2, out3, rtol=1e-4, atol=1e-4)


# the complex case for monarch
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_diag_butterfly_multiply_reference_rectangular(device, dtype):
    # set seed
    torch.random.manual_seed(0)
    n = 768
    batch_size = 3
    x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
    w1_bfly = torch.randn(
        8, 96 * 2, 96, device=x.device, dtype=x.dtype, requires_grad=True
    )  # （num_blocks, out_dim, in_dim)
    w2_bfly = torch.randn(24, 16, 64, device=x.device, dtype=x.dtype, requires_grad=True)

    out2 = blockdiag_butterfly_multiply_reference(
        x, w1_bfly, w2_bfly, version=2
    )  # (batch, n) -> (batch, m), where m = 24 * 16
    out3 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=3)
    assert torch.allclose(out2, out3, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("log_n", [4, 10, 12])
def test_block_diag_butterfly_multiply(log_n, device, dtype):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    batch_size = 3
    x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
    w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    pytest.set_trace()
    out = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly)
    grad = torch.randn_like(out)
    dx, dw1_bfly, dw2_bfly = torch.autograd.grad(out, (x, w1_bfly, w2_bfly), grad, retain_graph=True)
    assert out.shape == (batch_size, n)
    pytest.set_trace()
    out_ref = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly)
    dx_ref, dw1_bfly_ref, dw2_bfly_ref = torch.autograd.grad(out_ref, (x, w1_bfly, w2_bfly), grad, retain_graph=True)
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dx, dx_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dw1_bfly, dw1_bfly_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dw2_bfly, dw2_bfly_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_diag_butterfly_multiply_rectangular(device, dtype):
    # set seed
    torch.random.manual_seed(0)
    n = 768
    batch_size = 3
    x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
    w1_bfly = torch.randn(8, 96 * 2, 96, device=x.device, dtype=x.dtype, requires_grad=True)
    w2_bfly = torch.randn(24, 16, 64, device=x.device, dtype=x.dtype, requires_grad=True)
    out = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly)
    grad = torch.randn_like(out)
    dx, dw1_bfly, dw2_bfly = torch.autograd.grad(out, (x, w1_bfly, w2_bfly), grad, retain_graph=True)
    assert out.shape == (batch_size, 24 * 16)
    out_ref = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly)
    dx_ref, dw1_bfly_ref, dw2_bfly_ref = torch.autograd.grad(out_ref, (x, w1_bfly, w2_bfly), grad, retain_graph=True)
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dx, dx_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dw1_bfly, dw1_bfly_ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(dw2_bfly, dw2_bfly_ref, rtol=1e-4, atol=1e-4)
