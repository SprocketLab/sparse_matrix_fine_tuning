import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import autograd
from .activations import *    

# Modified from https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/bmm.py
# It won't really make sense to fuse activations here--they usually have different block dims.
# And for LoRA-style PEFT activation is applied after adding to main output.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def bmm_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_al,
    stride_am,
    stride_ak,
    stride_bl,
    stride_bk,
    stride_bn,
    stride_ol,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    GROUP_SIZE_M = 8

    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + pid_batch * stride_al)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn + pid_batch * stride_bl)

    o = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)

        x_ptrs += BLOCK_K * stride_ak
        y_ptrs += BLOCK_K * stride_bk

    if ACTIVATION is not None:
        o = ACTIVATION(o)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)
    

@triton.jit
def monarch_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_al,
    stride_am,
    stride_ak,
    stride_bl,
    stride_bk,
    stride_bn,
    stride_ol,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """ implement the following in a fused triton kernel using the bmm helper above
    def forward(ctx, x, w1_bfly, w2_bfly, out1, out2):
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.bmm(
            x_reshaped, w1_bfly.transpose(-1, -2), out=out1
        )
        out1 = (
            out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).transpose(0, 1).contiguous()
        )  # -> (batch_dim, k, q) -> (batch_dim, r, l) -> (batch_dim, l, r) -> (l, batch_dim, r)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)  # (l, batch_dim, r) @ (l, r, s) -> (l, batch_dim, s)
        out2 = out2.permute(1, 2, 0).reshape(
            *batch_shape, s * l
        )  # (batch_dim, l, s) -> (batch_dim, s, l) -> (batch_dim, m = s * l)
        return out2 
    """
    x_reshaped  = tl.reshape(x_ptr, (M, K))
    
    

def monarch_forward(x, y, activation=None, out=None):
    B, SEQ_LEN, K = x.shape
    if y.ndim == 2:
        y = y.unsqueeze(0).expand(B, -1, -1)

    _, K, N = y.shape
    # assert (K % 32 == 0), f"K must be divisible by 32"
    if out is None or out.shape != (B, SEQ_LEN, N):
        out = torch.empty((B, SEQ_LEN, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        B,
        triton.cdiv(SEQ_LEN, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch.cuda.device(x.device):
        bmm_kernel[grid](
            x,
            y,
            out,
            SEQ_LEN,
            N,
            K,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            ACTIVATION=activation,
        )
    return out




class _relu_squared(autograd.Function):
    @classmethod
    def forward(self, ctx, x, w):
        o = monarch_forward(x, w, activation=relu_squared_activation)
        if x.requires_grad:
            ctx.save_for_backward(x, w, o)
        return o

    @classmethod
    def backward(self, ctx, dy):
        x, w, o = ctx.saved_tensors
        dy = torch.sqrt(o) * 2 * dy
        dx = monarch_forward(dy, w.t())
        dw = monarch_forward(x.transpose(-1, -2), dy)
        return dx, dw


triton_relu_squared = _relu_squared.apply


def fused_relu_squared(x, w, use_triton=False):
    if use_triton:
        return triton_relu_squared(x, w)

    return F.relu(x @ w) ** 2
