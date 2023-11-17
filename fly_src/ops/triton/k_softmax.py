# Copied from https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_softmax.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
import triton.language as tl

# CREDITS: This is adapted from the vanilla Triton example. See https://openai.com/blog/triton/
# and https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html


def get_depth(K):
    return triton.next_power_of_2(K)


# autotune: Triton will test out these configurations, and automatically pick the fastest one.
# heuristic: add arguments to the kernel call automatically given some heuristics.
# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['Y'].dtype == torch.float16})
@triton.jit
def _softmax(
    Y, X, M,
    stride_ym, stride_yn,
    stride_xm, stride_xn,
    stride_m,
    K,
    LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # fmt: om

    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, meaning that this is equivalent to torch.softmax(tensor, dim=-1)

    Note, if the last dimension is large, say 128K elements, the kernel compile time can shot up to many minutes when
    the kernel is run for the first time.
    """

    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, DEPTH)

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m * stride_xm + n * stride_xn + k

    # load input data; pad out-of-bounds elements with 0
    io_mask = k < K

    # Causal - 1: skip on the loads directly
    if CAUSAL:
        io_mask = io_mask & (k <= n)

    x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))

    # Causal - 2: enforce correctness over a couple of misloaded values
    if CAUSAL:
        off = float("-inf")
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)

    if MASK_TYPE is not None:
        if MASK_TYPE == 'qk':
            mask_ptrs = M + n * stride_m + k
        elif MASK_TYPE == 'bk':
            mask_ptrs = M + m * stride_m + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
        x += add_mask

    # compute numerically-stable softmax
    z = x - tl.max(x, axis=0)

    if IS_FP16:
        # tl.exp() crashes on fp16 values
        # See https://github.com/openai/triton/issues/241
        z = z.to(tl.float32)

    num = tl.exp(z)
    denom = tl.sum(num, axis=0)

    if LOG:
        y = z - tl.log(denom)
    else:
        y = num / denom

    # write back to Y.
    # we only write once, hence the "fused" softmax naming
    y_ptrs = Y + m * stride_ym + n * stride_yn + k

    # technically we could write only the lower triangular matrix in the causal case
    # but this is deemed to error prone
    tl.store(y_ptrs, y, mask=k < K)


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['GradIn'].dtype == torch.float16})
@triton.jit
def _softmax_backward(
    GradIn, GradOut, Out,
    stride_bm, stride_bn,
    stride_gm, stride_gn,
    stride_om, stride_on,
    K,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # fmt: on

    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """

    m = tl.program_id(0)
    n = tl.program_id(1)

    # col indices
    k = tl.arange(0, DEPTH)

    # the memory address of all the elements that we want to load can be computed as follows
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k

    # load input data; pad out-of-bounds elements with 0
    io_mask = k < K

    # Causal - 1: skip on the loads directly
    if CAUSAL:
        io_mask = io_mask & (k <= n)

    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))

    # Causal - 2: enforce correctness over a couple of misloaded values
    if CAUSAL:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)

    if LOG:
        s = tl.sum(g, 0)
        if IS_FP16:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        # Step 1: Compute the intermediate sum used for the gradient
        s = tl.sum(g * o, 0)

        # Step 2: Compute the gradients
        grad_in = o * (g - s)

    # write back to the input gradients
    # technically we could write only the lower triangular matrix in the causal case
    # but this is deemed to error prone
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)
