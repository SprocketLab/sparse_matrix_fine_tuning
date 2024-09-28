import torch
import triton
import triton.language as tl


# Modified from https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/bmm.py
# It won't really make sense to fuse activations here--they usually have different block dims.
# And for LoRA-style PEFT activation is applied after adding to main output.
# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
#         ),
#         triton.Config(
#             {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8
#         ),
#         triton.Config(
#             {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
#         ),
#         triton.Config(
#             {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
#         ),
#         triton.Config(
#             {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
#         ),
#         triton.Config(
#             {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
#         ),
#         triton.Config(
#             {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
#         ),
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def bmm(
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
    GROUP_SIZE_M: tl.constexpr = 8,
):
    o = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)

        BLOCK_K * stride_ak
        BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)


@triton.jit
def monarch_forward(
    x_ptr,
    o_ptr1,
    o_ptr2,
    w1_bfly_ptr,
    w2_bfly_ptr,
    M,
    N,
    K,
    stride_xl,
    stride_xm,
    stride_xk,
    stride_wl,
    stride_wk,
    stride_wn,
    stride_ol,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """implement the following in a fused triton kernel using the bmm helper above
    def forward(ctx, x, w1_bfly, w2_bfly, out1, out2):
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        x_reshaped = x.reshape(seq_dim, k, p).transpose(0, 1)
        out1 = torch.bmm(
            x_reshaped, w1_bfly.transpose(-1, -2), out=out1
        )
        out1 = (
            out1.transpose(0, 1).reshape(seq_dim, r, l).transpose(-1, -2).transpose(0, 1).contiguous()
        )  # -> (seq_dim, k, q) -> (seq_dim, r, l) -> (seq_dim, l, r) -> (l, seq_dim, r)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)  # (l, seq_dim, r) @ (l, r, s) -> (l, seq_dim, s)
        out2 = out2.permute(1, 2, 0).reshape(
            *batch_shape, s * l
        )  # (seq_dim, l, s) -> (seq_dim, s, l) -> (seq_dim, m = s * l)
        return out2
    """

    # Grouped ordering for better l2 reuse
    tl.program_id(0)
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
    # x_ptrs = x_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + pid_batch * stride_al)
    # y_ptrs = y_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn + pid_batch * stride_bl)
    x_ptrs = tl.make_block_ptr(
        x_ptr,
        shape=(M, K),  # the shape that's split over blocks
        strides=(stride_xm, stride_xk),
        offsets=(offs_am, offs_k),
        block_shape=(BLOCK_M, BLOCK_K),
    )
    w1_ptr = tl.make_block_ptr(
        w1_bfly_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(offs_k, offs_bn),
        block_shape=(BLOCK_K, BLOCK_N),
    )


@triton.jit
def monarch_backward():
    """Implements the following in triton:
        x, w1_bfly, w2_bfly, out1, *_ = ctx.saved_tensors
    batch_shape, n = x.shape[:-1], x.shape[-1]
    seq_dim = np.prod(batch_shape)
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    dx, dw1_bfly, dw2_bfly = None, None, None
    # dout_reshaped = dout.reshape(seq_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
    dout_reshaped = dout.reshape(seq_dim, s, l).transpose(-1, -2).contiguous()
    dout_reshaped = dout_reshaped.transpose(0, 1)
    if ctx.needs_input_grad[2]:
        # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
        # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
        dw2_bfly = torch.bmm(
            dout_reshaped.transpose(-1, -2), out1.conj()
        )  # (l, s, seq_dim) @ (l, seq_dim, r) -> (l, s, r)
    if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
        dout1 = torch.empty(seq_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
        dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
        dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(seq_dim, k, q).transpose(0, 1)
        # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(seq_dim, k, p, device=x.device, dtype=x.dtype)
            dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(seq_dim, k, p).transpose(0, 1)
            dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
    return dx, dw1_bfly, dw2_bfly, None, None
    """


class MonarchKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_bfly, w2_bfly, out=None):
        # For Llama 7B fine-tuning on Math, this is like (2, 286, 4096)
        B, SEQ_LEN, HID_DIM = x.shape
        seq_dim = B * SEQ_LEN
        nblocks1, q, p = w1_bfly.shape
        nblocks2, s, r = w2_bfly.shape

        out1 = torch.empty(nblocks1, seq_dim, q, device=x.device, dtype=x.dtype)
        out2 = torch.empty(nblocks2, seq_dim, s, device=x.device, dtype=x.dtype)

        grid = lambda META: (
            B,
            triton.cdiv(seq_dim, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        with torch.cuda.device(x.device):
            monarch_forward[grid](
                x,
                out1,
                out2,
                w1_bfly,
                w2_bfly,
                seq_dim,
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
            )
        ctx.save_for_backward(x, y, w1_bfly, w2_bfly, out1)
        return out

    @staticmethod
    def backward(ctx, dout):
        x,
