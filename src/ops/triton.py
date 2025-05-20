import numpy as np
import torch
import triton
import triton.language as tl


def config_gen():
    configs = []
    for BSEQ in [32, 64, 128]:
        for BK in [32, 64, 128]:
            for BN in [32, 64, 128]:
                for num_warps in [4, 8]:
                    for num_stages in [4, 5]:
                        if BSEQ * BK * BN <= 65536:
                            # This is the max number of elements that can be loaded in a single kernel
                            # (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
                            # num_stages = 4  # Even flash-attn has up to 5 stages (https://github.com/triton-lang/triton/blob/6af74b2f4535682abfc0b08958bc2c6831036d29/python/tutorials/06-fused-attention.py#L489)
                            # num_warps = 4
                            # Filter
                            if BK <= 64:
                                num_stages = 5
                                num_warps = 8
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_SEQ": BSEQ, "BLOCK_SIZE_K": BK, "BLOCK_SIZE_N": BN, "GROUP_SIZE_M": 8},
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                )
                            )
    return configs


# @triton.autotune(
#     config_gen(),
#     key=["N_BLK", "BLK1_IN", "BLK2_OUT"],
# )
@triton.jit
def monarch_backward(
    dout_ptr,
    out1_ptr,
    x_ptr,
    w1_bfly_ptr,
    w2_bfly_ptr,
    dx_ptr,
    dw1_bfly_ptr,
    dw2_bfly_ptr,
    SEQ_DIM: tl.constexpr,
    N_BLK: tl.constexpr,
    BLK1_IN: tl.constexpr,
    BLK1_OUT: tl.constexpr,
    BLK2_OUT: tl.constexpr,
    stride_dout_l, stride_dout_m, stride_dout_n,
    stride_out1_l, stride_out1_m, stride_out1_r,
    stride_xl, stride_xm, stride_xk,
    stride_w1l, stride_w1r, stride_w1k,
    stride_w2l, stride_w2n, stride_w2r,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_M: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 32,
    GROUP_SIZE_M: tl.constexpr = 8,
    BLOCK_SIZE_SEQ: tl.constexpr = 64,
):
    tl.static_assert(SEQ_DIM > 0, "SEQ_DIM must be positive")
    tl.static_assert(N_BLK > 0, "N_BLK must be positive")
    tl.static_assert(BLK2_OUT > 0, "BLK2_OUT must be positive")

    BLK2_IN: tl.constexpr = BLK1_OUT
    # pid_batch = tl.program_id(0)
    pid = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Compute grouped row ids for M x K x N
    num_pid_m = tl.cdiv(BLK2_OUT, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(BLK1_IN, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m_in_group = pid % GROUP_SIZE_M

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    offs_seq = pid_seq * BLOCK_SIZE_SEQ

    x_ptrs = tl.make_block_ptr(
        x_ptr,
        shape=(N_BLK, SEQ_DIM, BLK1_IN),
        strides=(stride_xl, stride_xm, stride_xk),
        offsets=(0, offs_seq, offs_n),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLOCK_SIZE_N),
        order=(2, 1, 0),
    )
    dx_ptrs = tl.make_block_ptr(
        dx_ptr,
        shape=(N_BLK, SEQ_DIM, BLK1_IN),
        strides=(stride_xl, stride_xm, stride_xk),
        offsets=(0, offs_seq, offs_n),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLOCK_SIZE_N),
        order=(2, 1, 0),
    )
    out1_ptrs = tl.make_block_ptr(
        out1_ptr,
        shape=(N_BLK, SEQ_DIM, BLK1_OUT),
        strides=(stride_out1_l, stride_out1_m, stride_out1_r),
        offsets=(0, offs_seq, 0),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLK1_OUT),
        order=(2, 1, 0),
    )
    dout_ptrs = tl.make_block_ptr(
        dout_ptr,
        shape=(N_BLK, SEQ_DIM, BLK2_OUT),
        strides=(stride_dout_l, stride_dout_m, stride_dout_n),
        offsets=(0, offs_seq, offs_m),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLOCK_SIZE_M),
        order=(2, 1, 0),
    )
    w1_ptrs = tl.make_block_ptr(
        w1_bfly_ptr,
        shape=(N_BLK, BLK1_OUT, BLK1_IN),
        strides=(stride_w1l, stride_w1r, stride_w1k),
        offsets=(0, 0, offs_n),
        block_shape=(N_BLK, BLK1_OUT, BLOCK_SIZE_N),
        order=(2, 1, 0),
    )
    w2_ptrs = tl.make_block_ptr(
        w2_bfly_ptr,
        shape=(N_BLK, BLK2_OUT, BLK2_IN),
        strides=(stride_w2l, stride_w2n, stride_w2r),
        offsets=(0, offs_m, 0),
        block_shape=(N_BLK, BLOCK_SIZE_M, BLK2_IN),
        order=(2, 1, 0),
    )
    # Replace atomic_add with normal pointers for dw1 (nblocks, blk2_in, seq_dim)
    base_dw1 = dw1_bfly_ptr #s+ pid * stride_w1l
    nblocks = tl.arange(0, N_BLK)[:, None, None]
    k_dw1 = tl.arange(0, BLK1_OUT)[None, :, None]
    j_dw1 = tl.arange(0, BLOCK_SIZE_N)[None, None, :]
    
    # Replace atomic_add with normal pointers for dw2 (nblocks2, blk2_out, blk2_in)
    w1_bfly = tl.load(w1_ptrs, boundary_check=(1, 2), eviction_policy="evict_last")
    out1 = tl.load(out1_ptrs, boundary_check=(1, 2), eviction_policy="evict_first")
    dout = tl.load(dout_ptrs, boundary_check=(1, 2), eviction_policy="evict_first")
    # Compute dw2
    # (nblocks2, blk2_out, seq_dim) @ (nblocks2, seq_dim, blk1_out) -> (nblocks2, blk2_out, blk1_out)
    dw2_bfly = tl.dot(dout.trans(0, 2, 1), out1)#, out_dtype=out1.dtype)
    if pid_n == 0:
        base_dw2 = tl.zeros((1, 1, 1), dtype=tl.int32) + dw2_bfly_ptr
        i = tl.arange(0, BLOCK_SIZE_M)[None, :, None]
        j = tl.arange(0, BLK2_IN)[None, None, :]
        offsets = nblocks * stride_w2l + (i + offs_m)  * stride_w2n + j * stride_w2r
        dw2_pointers = base_dw2 + offsets
        mask_dw2 = (offs_m + i < BLK2_OUT) & (j < BLK2_IN)
        tl.atomic_add(dw2_pointers, dw2_bfly.to(out1.dtype), mask=mask_dw2)

    # Split-k over block2 output dim, because we will use this intermediate result
    if pid_m_in_group == 0: 
        dout1 = tl.zeros((N_BLK, BLOCK_SIZE_SEQ, BLK2_IN), dtype=tl.float32)
        for _ in range(0, BLK2_OUT, BLOCK_SIZE_M):
            w2_bfly = tl.load(w2_ptrs, boundary_check=(1,), eviction_policy="evict_first")
            # (nblocks2, seq_dim, blk2_out) @ (nblocks2, blk2_out, blk2_in) -> (nblocks2, seq_dim, blk2_in)
            dout1 += tl.dot(dout, w2_bfly, out_dtype=tl.float32)
            w2_ptrs = tl.advance(w2_ptrs, (0, BLOCK_SIZE_M, 0))
            dout_ptrs = tl.advance(dout_ptrs, (0, 0, BLOCK_SIZE_M))            
            dout = tl.load(dout_ptrs, boundary_check=(1, 2), eviction_policy="evict_first")
        
    # Compute dx and dw1_bfly
        dout1 = dout1.to(dtype=out1.dtype)                         
        dout1 = dout1.trans(1, 2, 0)
        dout1 = dout1.reshape(BLOCK_SIZE_SEQ, N_BLK, BLK1_OUT)
        dout1 = dout1.trans(1, 0, 2)
        # (nblocks1, seq_dim, blk1_out) @ (nblocks1, blk1_out, blk1_in) -> (nblocks1, seq_dim, blk1_in)
        dx = tl.dot(dout1, w1_bfly)
        dx = dx.to(dtype=out1.dtype)
        tl.store(dx_ptrs, dx, boundary_check=(1, 2))
        
        x = tl.load(x_ptrs, boundary_check=(1, 2), eviction_policy="evict_first")
        # （nblocks2, blk2_in, seq_dim) @ (nblocks2, seq_dim, blk1_in) -> (nblocks2, blk2_in, blk1_in)
        dw1_bfly = tl.dot(tl.trans(dout1, 0, 2, 1), x, out_dtype=dout1.dtype)
        # Atomic add to global mem
        offs_n_current = offs_n
        offsets_dw1 = nblocks * stride_w1l + k_dw1 * stride_w1r + (offs_n_current + j_dw1) * stride_w1k
        dw1_pointers = base_dw1 + offsets_dw1
        mask_dw1 = (offs_n_current + j_dw1) < BLK1_IN
        tl.atomic_add(dw1_pointers, dw1_bfly, mask=mask_dw1)
            
# fmt: off
# Autotune can even make the kernel slower...
# @triton.autotune(
#     config_gen(),
#     key=["N_BLK", "BLK1_IN", "BLK2_OUT"],
#     rep=80,
#     warmup=15
# )
@triton.jit
def monarch_forward(
    x_ptr, o_ptr1, o_ptr2, w1_bfly_ptr, w2_bfly_ptr,
    SEQ_DIM,
    N_BLK:tl.constexpr,
    BLK1_IN:tl.constexpr,
    BLK1_OUT: tl.constexpr,
    BLK2_OUT: tl.constexpr,
    stride_xl, stride_xm, stride_xk,
    stride_w1l, stride_w1r, stride_w1k,
    stride_w2l, stride_w2n, stride_w2r,
    stride_o1l, stride_o1m, stride_o1k,
    stride_o2l, stride_o2m, stride_o2n,
    BLOCK_SIZE_SEQ: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 32,
    GROUP_SIZE_SEQ: tl.constexpr = 8,
):
    # fmt: on
    """
    Implements fused monarch forward as in `BlockdiagButterflyMultiply`.
    Each kernel comsumes (BLOCK_SEQ, N_BLK, BLK1_IN) elements in x, as we need to shuffle (transpose)
    the features in out1.
    """
    BLK2_IN: tl.constexpr = BLK1_OUT  # This is the block rank (usually small, e.g. 4 for PEFT)

    pid = tl.program_id(0)
    
    # Grouped ordering for better l2 cache reuse
    num_pid_m = tl.cdiv(SEQ_DIM, BLOCK_SIZE_SEQ)
    num_pid_n = tl.cdiv(BLK2_OUT, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_SEQ * num_pid_n
    group_id = pid // num_pid_in_group

    first_pid_m = group_id * GROUP_SIZE_SEQ
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_SEQ)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_SEQ
    offs_bn = pid_n * BLOCK_SIZE_N

    x_ptrs = tl.make_block_ptr(
        x_ptr,
        shape=(N_BLK, SEQ_DIM, BLK1_IN),  # the full shape that's split over blocks
        strides=(stride_xl, stride_xm, stride_xk),
        offsets=(0, offs_am, 0),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLOCK_SIZE_K), # block_shape requires List[tl.constexpr]
        order=(2, 1, 0)
    )

    # TODO: load w in transposed order instead of in-kernel transpose?
    w1_ptrs = tl.make_block_ptr(
        w1_bfly_ptr,
        shape=(N_BLK, BLK1_OUT, BLK1_IN),
        strides=(stride_w1l, stride_w1r, stride_w1k),
        offsets=(0, 0, 0),
        block_shape=(N_BLK, BLK1_OUT, BLOCK_SIZE_K),
        order=(2, 1, 0),
    )

    w2_ptrs = tl.make_block_ptr(
        w2_bfly_ptr,
        shape=(N_BLK, BLK2_OUT, BLK2_IN),
        strides=(stride_w2l, stride_w2n, stride_w2r),
        offsets=(0, offs_bn, 0),
        block_shape=(N_BLK, BLOCK_SIZE_N, BLK2_IN),
        order=(2, 1, 0),
    )
    out1_ptrs = tl.make_block_ptr(
        o_ptr1,
        shape=(N_BLK, SEQ_DIM, BLK1_OUT),
        strides=(stride_o1l, stride_o1m, stride_o1k),
        offsets=(0, offs_am, 0),
        block_shape=(N_BLK, BLOCK_SIZE_SEQ, BLK1_OUT),
        order=(2, 1, 0),
    )

    out2_ptrs = tl.make_block_ptr(
        o_ptr2,
        shape=(SEQ_DIM, BLK2_OUT, N_BLK),
        strides=(stride_o2l, stride_o2m, stride_o2n),
        offsets=(offs_am, offs_bn, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_N, N_BLK),
        order=(2, 1, 0),
    )

    x = tl.load(x_ptrs, boundary_check=(1, 2), eviction_policy="evict_first", padding_option="zero")  # Prefetch
    dtype = x.dtype  
    out1 = tl.zeros((N_BLK, BLOCK_SIZE_SEQ, BLK1_OUT), dtype=tl.float32) 
    # Do NOT use split-k + atomic here, which requires a round trip to HBM and cross-block sync
    for k in range(0, BLK1_IN, BLOCK_SIZE_K):
        w1_bfly = tl.load(w1_ptrs, boundary_check=(2, ), eviction_policy="evict_first", padding_option="zero").to(dtype)
        w1_bfly = tl.trans(w1_bfly, 0, 2, 1)  # -> (n_blk, blk1_in, blk1_out)
        out1 += tl.dot(
            x, w1_bfly
        )  # (n_blk, seq_dim, blk1_in) @ (n_blk, blk1_in, blk1_out) -> (n_blk, seq_dim, blk1_out).

        x_ptrs = tl.advance(x_ptrs, (0, 0, BLOCK_SIZE_K))
        w1_ptrs = tl.advance(w1_ptrs, (0, 0, BLOCK_SIZE_K))
        # Prefetch
        x = tl.load(x_ptrs, boundary_check=(1, 2), eviction_policy="evict_first", padding_option="zero")

    # shuffle features
    out1 = tl.trans(out1, (1, 0, 2)) # -> (seq_dim, n_blk, blk1_out)
    out1 = tl.reshape(out1, (BLOCK_SIZE_SEQ, BLK2_IN, N_BLK))
    out1 = tl.trans(out1, (2, 0, 1)).to(dtype) # -> (n_blk, seq_dim, blk2_in)
    tl.store(out1_ptrs, out1, boundary_check=(1, ))

    w2_bfly = tl.load(w2_ptrs, boundary_check=(1, 2), padding_option="zero").to(dtype)
    w2_bfly = tl.trans(w2_bfly, 0, 2, 1)  # -> (blk2_in, blk2_out)
    out2 = tl.dot(out1, w2_bfly, out_dtype=dtype)  # (n_blk, seq_dim, blk1_out) @ (n_blk, blk2_in, blk2_out) -> (n_blk, seq_dim, blk2_out)
    out2 = tl.trans(out2, 1, 2, 0)  # -> (seq_dim, blk2_out, n_blk)
    tl.store(out2_ptrs, out2, boundary_check=(0, 1))



class MonarchKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1_bfly, w2_bfly, debug_out1=False):
        # For Llama 7B fine-tuning on Math, this is like (2, 286, 4096)
        BATCH_SHAPE, HID_DIM = x.shape
        seq_dim = int(np.prod(BATCH_SHAPE))

        nblocks1, blk1_out, blk1_in = w1_bfly.shape
        nblocks2, blk2_out, blk2_in = w2_bfly.shape
        assert nblocks1 == nblocks2 and blk1_out == blk2_in, "Doesn't support irregular blocks yet"
        nblocks = nblocks1
        assert nblocks * blk1_in == HID_DIM
        assert nblocks * blk2_in == nblocks * blk1_out

        # TODO: eliminate overhead of contiguous by in-kernel transpose?
        x_reshaped = x.view(seq_dim, HID_DIM).view(seq_dim, nblocks1, blk1_in).transpose(0, 1).contiguous()
        out1 = torch.empty(nblocks, seq_dim, blk1_out, device=x.device, dtype=x.dtype)
        out2 = torch.empty(seq_dim, blk2_out, nblocks, device=x.device, dtype=x.dtype)
        # For a monarch input (nblocks1, seq_dim, blk1_in) reshaped from (seq_dim, in_dim),
        # (like (M, K) @ (K, N) in normal matmul) we wanna parallelize over in_dim and seq_dim.
        # seq_dim for math and Alpaca tuning is small; hidden_dim is 4096 for Llama 7B. e.g. (4, 666, 1024)
        # We launch a 2d grid of blocks sized (1, BLOCK_SIZE_SEQ * BLOCK_SIZE_N), fusing the batch dim together.
        grid = lambda META: (
            triton.cdiv(seq_dim, META["BLOCK_SIZE_SEQ"]) * triton.cdiv(blk2_out, META["BLOCK_SIZE_N"]), 
        )
        # fmt: off
        monarch_forward[grid](
            x_reshaped, out1, out2, w1_bfly, w2_bfly,
            seq_dim, nblocks, blk1_in, blk1_out, blk2_out,
            x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2),
            w1_bfly.stride(0), w1_bfly.stride(1), w1_bfly.stride(2),
            w2_bfly.stride(0), w2_bfly.stride(1), w2_bfly.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            out2.stride(0), out2.stride(1), out2.stride(2),
            num_warps=8, 
        )

        # fmt: on
        out2 = out2.view(BATCH_SHAPE, nblocks * blk2_out)

        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        if debug_out1:
            return out2, out1
        return out2

    @staticmethod
    def backward(ctx, dout, *args):
        x, w1_bfly, w2_bfly, out1, *_ = ctx.saved_tensors
        BATCH_SHAPE, HID_DIM = x.shape
        seq_dim = int(np.prod(BATCH_SHAPE))
        nblocks1, blk1_out, blk1_in = w1_bfly.shape
        nblocks2, blk2_out, blk2_in = w2_bfly.shape
        assert nblocks1 == nblocks2 and blk1_out == blk2_in, "Doesn't support irregular blocks yet"
        nblocks = nblocks1 = nblocks2

        # (nblocks2, seq_dim, blk1_out)
        x = x.view(seq_dim, HID_DIM).view(seq_dim, nblocks1, blk1_in).transpose(0, 1).contiguous()

        # Allocate buffers
        # (nblocks2, seq_dim, blk2_out)
        dout = dout.view(seq_dim, blk2_out, nblocks2).permute(2, 0, 1).contiguous()
        dw1_bfly = torch.zeros(nblocks1, blk2_in, blk1_in, device=w1_bfly.device, dtype=w1_bfly.dtype)
        dw2_bfly = torch.zeros(nblocks2, blk2_out, blk2_in, device=w2_bfly.device, dtype=w2_bfly.dtype)
        dx = torch.zeros(nblocks1, seq_dim, blk1_in, device=x.device, dtype=x.dtype)
        # Triton doesn't have conjugate?
        w1_bfly = w1_bfly.conj()
        w2_bfly = w2_bfly.conj()
        out1 = out1.conj()  # (nblocks1, seq_dim, blk1_out)
        x = x.conj()

        grid = lambda META: (
            triton.cdiv(blk2_out, META["BLOCK_SIZE_M"]) * triton.cdiv(blk1_in, META["BLOCK_SIZE_N"]),
            triton.cdiv(seq_dim, META["BLOCK_SIZE_SEQ"])
        )
        # dw2_bfly = dw2_bfly if ctx.needs_input_grad[2] else None
        # dx = dx if ctx.needs_input_grad[0] else None
        # dw1_bfly = dw1_bfly if ctx.needs_input_grad[1] else None
        # fmt: off
        with torch.cuda.device(x.device):
            monarch_backward[grid](
                dout, out1, x, w1_bfly, w2_bfly,
                dx, dw1_bfly, dw2_bfly,
                seq_dim, nblocks, blk1_in, blk1_out, blk2_out,
                dout.stride(0), dout.stride(1), dout.stride(2),
                out1.stride(0), out1.stride(1), out1.stride(2),
                x.stride(0), x.stride(1), x.stride(2),
                w1_bfly.stride(0), w1_bfly.stride(1), w1_bfly.stride(2),
                w2_bfly.stride(0), w2_bfly.stride(1), w2_bfly.stride(2),
            )
        # fmt: on
        dx = dx.reshape(BATCH_SHAPE, HID_DIM)
        return dx, dw1_bfly, dw2_bfly, None


monarch_kernel = MonarchKernel.apply
