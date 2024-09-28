import math

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F


class BlockdiagMultiply(torch.autograd.Function):
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd()
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        x_reshaped = x.reshape(seq_dim, nblocks, p).transpose(0, 1)  # (nblocks, seq_dim, p)

        out = torch.empty(nblocks, seq_dim, q, device=x.device, dtype=x.dtype)
        out = torch.bmm(x_reshaped, weight.transpose(-1, -2), out=out).transpose(
            0, 1
        )  # (nblocks, seq_dim, blk_sz) @ (nblocks, blk_sz, blk_r) -> (nblocks, seq_dim, q)
        return out.reshape(*batch_shape, nblocks * q)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        dx, dweight = None, None
        dout_reshaped = dout.reshape(seq_dim, nblocks, q).transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(seq_dim, nblocks, p, device=x.device, dtype=x.dtype)
            dx = (
                torch.bmm(dout_reshaped, weight.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            )
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(seq_dim, nblocks, p).transpose(0, 1)
            dweight = torch.bmm(dout_reshaped.transpose(-1, -2), x_reshaped.conj())
        return dx, dweight


single_monarch_mult = BlockdiagMultiply.apply


class BlockdiagButterflyMultiply(torch.autograd.Function):
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (nblocks, blk_r, blk_sz)
        w2_bfly: (nblocks, blk_sz, blk_r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly, out1=None, out2=None):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)

        w1_bfly = w1_bfly.to(x.dtype)
        w2_bfly = w2_bfly.to(x.dtype)

        nblocks1, q, p = w1_bfly.shape
        nblocks2, s, r = w2_bfly.shape
        assert nblocks1 * p == n
        assert nblocks2 * r == nblocks1 * q
        # For Llama 7B on Math x_reshaped is like (4, 666, 1024)
        # For instruction tuning seq dim is around 150
        x_reshaped = x.reshape(seq_dim, nblocks1, p).transpose(0, 1)
        out1 = torch.empty(nblocks1, seq_dim, q, device=x.device, dtype=x.dtype)
        out1 = torch.bmm(
            x_reshaped, w1_bfly.transpose(-1, -2), out=out1
        )  # (nblocks1, seq_dim, p) @ (nblocks1, p, q) -> (nblocks1, seq_dim, q)
        del x_reshaped

        # out1 = out1.transpose(0, 1).reshape(seq_dim, r, nblocks2).transpose(-1, -2).contiguous().transpose(0, 1)
        out1 = (
            out1.transpose(0, 1).reshape(seq_dim, r, nblocks2).transpose(-1, -2).transpose(0, 1)
        )  # -> (seq_dim, nblocks1, q) -> (seq_dim, r, nblocks2) -> (seq_dim, nblocks2, r) -> (nblocks2, seq_dim, r)

        out2 = torch.empty(nblocks2, seq_dim, s, device=x.device, dtype=x.dtype)
        out2 = torch.bmm(
            out1, w2_bfly.transpose(-1, -2), out=out2
        )  # (nblocks2, seq_dim, r) @ (nblocks2, r, s) -> (nblocks2, seq_dim, s)

        out2 = out2.permute(1, 2, 0).reshape(
            *batch_shape, s * nblocks2
        )  # (seq_dim, nblocks2, s) -> (seq_dim, s, nblocks2) -> (seq_dim, m = s * nblocks2)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1, None, None)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1, *_ = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)
        nblocks1, q, p = w1_bfly.shape
        nblocks2, s, r = w2_bfly.shape

        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(seq_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(seq_dim, s, nblocks2).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(nblocks2, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(
                dout_reshaped.transpose(-1, -2), out1.conj()
            )  # (nblocks2, s, seq_dim) @ (nblocks2, seq_dim, r) -> (nblocks2, s, r)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(seq_dim, nblocks2, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(seq_dim, nblocks1, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(seq_dim, nblocks1, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(seq_dim, nblocks1, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly, None, None


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply


# Supports rectangular matrices
def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    (q, p) and (s, r) are blocks in the monarch matrix
    Arguments:
        x: (batch, n)

        Assume we project dense W (m, n) to w1_bfly and w2_bfly.
        w1_bfly: (nblocks1, q, p), NOTE where n = k * p = in_dim,  k = num_blocks, q = intermediate_out_dim, p = block1_in_dim
        NOTE Both q and p can be called "block size", and generally you can set q = p = sqrt(n).
        w2_bfly: (nblocks2, s, r), where l = k * q / r = n * q / (p * r)
        NOTE l * s = m, l = num_blocks, r = block2_in_dim, s = blk_out_dim
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError("version must be either 1, 2, or 3")
    batch, n = x.shape
    nblocks1, q, p = w1_bfly.shape
    nblocks2, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, "b (k p) -> b k p", k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum("bkp,kqp,qlk->blq", x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum("kqp,bkp->bkq", w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, "b k q -> b (k q)"), "b (r nblocks2) -> b l r", l=nblocks2)
        return torch.einsum("lsr,blr->bsl", w2_bfly, out1).reshape(batch, s * nblocks2)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = F.linear(x, w1_dense)
        out1 = rearrange(out1, "b (r nblocks2) -> b (l r)", l=nblocks2)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = F.linear(out1, w2_dense)
        out2 = rearrange(out2, "b (l s) -> b (s nblocks2)", l=nblocks2)
        return out2
