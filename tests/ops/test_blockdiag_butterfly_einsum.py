import math

import torch
import pytest
import os
import sys
# two dirs back 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply_reference
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.blockdiag_butterfly_einsum import (
    blockdiag_butterfly_multiply_einsum_simple, blockdiag_butterfly_project_einsum_simple,
    blockdiag_butterfly_multiply_einsum, blockdiag_butterfly_project_einsum,
    blockdiag_butterfly_multiply_einsum_rank, blockdiag_butterfly_project_einsum_rank
)


# @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# @pytest.mark.parametrize('log_n', [4, 10, 12])
# def test_block_diag_butterfly_multiply_einsum_simple(log_n, device, dtype):
#     # set seed
#     torch.random.manual_seed(0)
#     n = 1 << log_n
#     sqrtn = 1 << (log_n // 2)
#     batch_size = 3
#     x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
#     w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
#     w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
#     out1 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2)
#     out2 = blockdiag_butterfly_multiply_einsum_simple(x, w1_bfly, w2_bfly)
#     assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-4)


# @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_block_diag_butterfly_multiply_einsum_simple_rectangular(device, dtype):
#     # set seed
#     torch.random.manual_seed(0)
#     n = 768
#     batch_size = 3
#     x = torch.randn(batch_size, n, device=device, dtype=dtype, requires_grad=True)
#     w1_bfly = torch.randn(8, 96 * 2, 96, device=x.device, dtype=x.dtype, requires_grad=True)
#     w2_bfly = torch.randn(96 * 2, 16, 8, device=x.device, dtype=x.dtype, requires_grad=True)
#     out2 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2)
#     out3 = blockdiag_butterfly_multiply_einsum_simple(x, w1_bfly, w2_bfly)
#     assert torch.allclose(out2, out3, rtol=1e-4, atol=1e-4)


# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# @pytest.mark.parametrize('log_n', [2, 4, 10])
# def test_block_diag_butterfly_project_einsum_sqrtn(log_n, device):
#     # set seed
#     torch.random.manual_seed(0)
#     n = 1 << log_n
#     sqrtn = 1 << (log_n // 2)
#     x = torch.eye(n, device=device)
#     w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
#     w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
#     bfly = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly).t()
#     w1_bfly_projected, w2_bfly_projected = blockdiag_butterfly_project_einsum_simple(bfly,
#                                                                                      nblocks1=sqrtn,
#                                                                                      nblocks2=sqrtn)
#     bfly_projected = blockdiag_butterfly_multiply(x, w1_bfly_projected, w2_bfly_projected).t()
#     print((bfly_projected - bfly).abs().max())
#     assert torch.allclose(bfly_projected, bfly, rtol=1e-4, atol=1e-4)

# # @Wenxuan: this gives the monarch matrix decomposition and composition
# # TODO why is the erorr so low with rank 1 approximation?
# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_block_diag_butterfly_project_einsum_simple(device):
#     # set seed
#     torch.random.manual_seed(0)
#     n = 768
#     x = torch.eye(n, device=device)
#     nblocks1, nblocks2 = 8, 96 * 2
#     w1_bfly = torch.randn(nblocks1, nblocks2, 96, device=x.device, dtype=x.dtype, requires_grad=True)
#     w2_bfly = torch.randn(nblocks2, 16, nblocks1, device=x.device, dtype=x.dtype, requires_grad=True)
#     bfly = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly).t() # x is the identity, so bfly is just w1_bfly * w2_bfly---the monarch matrix!
#     w1_bfly_projected, w2_bfly_projected = blockdiag_butterfly_project_einsum_simple(bfly,
#                                                                                      nblocks1=nblocks1,
#                                                                                      nblocks2=nblocks2)
#     assert w1_bfly_projected.shape == w1_bfly.shape
#     assert w2_bfly_projected.shape == w2_bfly.shape
#     bfly_projected = blockdiag_butterfly_multiply(x, w1_bfly_projected, w2_bfly_projected).t()
#     print((bfly_projected - bfly).abs().max())
#     assert torch.allclose(bfly_projected, bfly, rtol=1e-4, atol=1e-4)


# @pytest.mark.parametrize('device', ['cpu', 'cuda'])
# def test_block_diag_butterfly_project_einsum(device):
#     # set seed
#     torch.random.manual_seed(0)
#     n = 768
#     x = torch.eye(n, device=device)
#     nblocks1, nblocks2 = 8, 24
#     b1, b2 = 8, 2
#     w1_bfly = torch.randn(nblocks1, nblocks2 * b1, 96, device=x.device, dtype=x.dtype,
#                           requires_grad=True)
#     w2_bfly = torch.randn(nblocks2, 16, nblocks1 * b1, device=x.device, dtype=x.dtype,
#                           requires_grad=True)
#     bfly = blockdiag_butterfly_multiply_einsum(x, w1_bfly, w2_bfly, b2=b2).t()
#     w1_bfly_projected, w2_bfly_projected = blockdiag_butterfly_project_einsum(bfly,
#                                                                               nblocks1=nblocks1,
#                                                                               nblocks2=nblocks2,
#                                                                               b1=b1, b2=b2)
#     assert w1_bfly_projected.shape == w1_bfly.shape
#     assert w2_bfly_projected.shape == w2_bfly.shape
#     bfly_projected = blockdiag_butterfly_multiply_einsum(x, w1_bfly_projected, w2_bfly_projected,
#                                                          b2=b2).t()
#     print((bfly_projected - bfly).abs().max())
#     assert torch.allclose(bfly_projected, bfly, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_block_diag_butterfly_project_einsum_rank(device):
    # set seed
    torch.random.manual_seed(0)
    n = 768
    x = torch.eye(n, device=device)
    nblocks1, nblocks2 = 4, 4
    rank = 8
    w1_bfly = torch.randn(nblocks1, nblocks2 * rank, n // nblocks1, device=x.device, dtype=x.dtype,
                          requires_grad=True)
    w2_bfly = torch.randn(nblocks2, n // nblocks2, nblocks1 * rank, device=x.device, dtype=x.dtype,
                          requires_grad=True)
    bfly = blockdiag_butterfly_multiply_einsum_rank(x, w1_bfly, w2_bfly).t()
    w1_bfly_projected, w2_bfly_projected, w1_rev, w2_rev = blockdiag_butterfly_project_einsum_rank(bfly,
                                                                                   nblocks1=nblocks1,
                                                                                   nblocks2=nblocks2,
                                                                                   rank=rank, reverse=True
                                                                                   )
    # assert w1_bfly_projected.shape == w1_bfly.shape
    # assert w2_bfly_projected.shape == w2_bfly.shape
    bfly_projected = bfly - blockdiag_butterfly_multiply_einsum_rank(x, w1_bfly_projected,
                                                              w2_bfly_projected).t()
    bfly_rev = blockdiag_butterfly_multiply_einsum_rank(x, w1_rev, w2_rev).t()
    print((bfly_projected - bfly_rev).abs().mean())
    torch.testing.assert_close(bfly_projected, bfly_rev, rtol=1e-4, atol=1e-4)
    assert False
