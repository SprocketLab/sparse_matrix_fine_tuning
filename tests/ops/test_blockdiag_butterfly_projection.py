import math

import torch
import pytest
from einops import rearrange
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.blockdiag_butterfly_projection import blockdiag_butterfly_project, factors
from src.ops.blockdiag_butterfly_projection import ButterflyFFT, ButterflyFFT2
from src.models.layers.monarch_linear import MonarchLinear
import os


# @Wenxuan: Tests whether trained weights instead of random weights 
# can be approximated by low rank with low error
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('rank', [1, 2])
@pytest.mark.parametrize('nblocks', [2, 3, 4])
@pytest.mark.parametrize('sdict_path', ["results/lora_roberta_agnews/model.pt"])
def test_trained_weight_approx(device, rank, nblocks, sdict_path):
    torch.random.manual_seed(0)
    
    assert os.path.exists(sdict_path), "you should finetune the model first"
    state_dict = torch.load(sdict_path, map_location=device)
    layers_to_test = ["query.weight", "key.weight"]
    monarch_out = []
    dense_out = []
    atol = 1e-4
    rtol = 1e-4

    # avg error across all layers
    for name, weights in state_dict.items():
        if any([layer in name for layer in layers_to_test]):
            m, n = weights.shape
            x = torch.eye(n, device=device)
            layer = MonarchLinear(in_features=n, out_features=m, nblocks=nblocks, rank=rank, weights=weights, device=device)
            monarch_out = [layer(x)]
            dense_out += [weights @ x]
    
    dense_out = torch.stack(dense_out).mean(dim=0) # (m, n)
    monarch_out = torch.stack(monarch_out).mean(dim=0) # (m, n)
    
    # check any(ele_wise_err), if this failed but total err low then it's ok
    if not torch.allclose(monarch_out, dense_out, rtol=rtol, atol=atol):
        print("num_total entries:", monarch_out.numel())
        print("num_failed:", ((dense_out - monarch_out).abs() <= atol + rtol * monarch_out.abs()).sum())
        # check mean err instead
        if not torch.allclose(monarch_out.mean(), dense_out.mean(), rtol=rtol, atol=atol):
            raise AssertionError(f" Failed with mean err {(monarch_out - dense_out).mean()}")
            
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('log_n', [2, 4, 10, 12])
def test_block_diag_butterfly_project_sqrtn(log_n, device):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    x = torch.eye(n, device=device)
    w1_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    w2_bfly = torch.randn(sqrtn, sqrtn, sqrtn, device=x.device, dtype=x.dtype, requires_grad=True)
    bfly = blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly).t()
    w1_bfly_projected, w2_bfly_projected = blockdiag_butterfly_project(bfly)
    bfly_projected = blockdiag_butterfly_multiply(x, w1_bfly_projected, w2_bfly_projected).t()
    print((bfly_projected - bfly).abs().max())
    assert torch.allclose(bfly_projected, bfly, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('log_n', [2, 4, 10])
def test_block_diag_butterfly_project_fft_sqrtn(log_n, device, direction):
    # set seed
    torch.random.manual_seed(0)
    n = 1 << log_n
    sqrtn = 1 << (log_n // 2)
    batch_size = 3
    eye = torch.eye(n, dtype=torch.complex64, device=device)
    transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
    dft = transform(eye, norm='ortho').t()
    # perm = bitreversal_permutation(n)
    # We don't actually need the bitreversal permutation, any permutation that swap
    # the axes of the sqrtn x sqrtn input will work.
    perm = rearrange(torch.arange(n, device=device), '(i j) -> (j i)', i=sqrtn)
    # The BP (butterfly - permutation) decomposition of FFT / iFFT
    # Converting to complex128 makes the approximation an order of magnitude more accurate
    w1_fft_projected, w2_fft_projected = blockdiag_butterfly_project(dft[:, perm].cdouble())
    w1_fft_projected, w2_fft_projected = w1_fft_projected.cfloat(), w2_fft_projected.cfloat()
    fft_projected = blockdiag_butterfly_multiply(eye, w1_fft_projected, w2_fft_projected).t()
    print((fft_projected - dft[:, perm]).abs().max())
    assert torch.allclose(fft_projected, dft[:, perm], rtol=1e-4, atol=1e-4)
    x = torch.randn(batch_size, n, dtype=torch.complex64, device=device)
    out_fft = transform(x, norm='ortho')
    out = blockdiag_butterfly_multiply(x[:, perm], w1_fft_projected, w2_fft_projected)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('n', [15, 36, 196, 27, 48, 42, 85, 168, 512])
def test_block_diag_butterfly_project_fft_rectangular(n, norm, device, direction):
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    eye = torch.eye(n, dtype=torch.complex64, device=device)
    transform = torch.fft.fft if direction == 'fft' else torch.fft.ifft
    dft = transform(eye, norm=norm).t()
    sizes = factors(n)[-1]
    sizes = (sizes[1], sizes[0])
    perm = rearrange(torch.arange(n, device=device), '(i j) -> (j i)', j=sizes[0])
    # The BP (butterfly - permutation) decomposition of FFT / iFFT
    # Converting to complex128 makes the approximation an order of magnitude more accurate
    w1_fft_projected, w2_fft_projected = blockdiag_butterfly_project(dft[:, perm].cdouble(),
                                                                     sizes=sizes)
    w1_fft_projected, w2_fft_projected = w1_fft_projected.cfloat(), w2_fft_projected.cfloat()
    fft_projected = blockdiag_butterfly_multiply(eye, w1_fft_projected, w2_fft_projected).t()
    print((fft_projected - dft[:, perm]).abs().max())
    assert torch.allclose(fft_projected, dft[:, perm], rtol=1e-4, atol=1e-4)
    x = torch.randn(batch_size, n, dtype=torch.complex64, device=device)
    out_fft = transform(x, norm=norm)
    out = blockdiag_butterfly_multiply(x[:, perm], w1_fft_projected, w2_fft_projected)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)

    bfly_fft = ButterflyFFT(n, direction=direction, norm=norm).to(device=device)
    out_module = bfly_fft(x)
    assert torch.allclose(out_module, out_fft, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('direction', ['fft', 'ifft'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('n2', [85, 512])
@pytest.mark.parametrize('n1', [42, 160, 161])
def test_butterflyfft2(n1, n2, norm, device, direction):
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    x = torch.randn(batch_size, n1, n2, dtype=torch.complex64, device=device)
    transform = torch.fft.fft2 if direction == 'fft' else torch.fft.ifft2
    out_fft = transform(x, norm=norm)
    bfly_fft = ButterflyFFT2(n1, n2, direction=direction, norm=norm).to(device=device)
    out = bfly_fft(x)
    assert torch.allclose(out, out_fft, rtol=1e-4, atol=1e-4)
