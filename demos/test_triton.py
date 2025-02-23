"""
python demos/test_triton.py -t True -b True
"""

import os
import sys

# os.environ["TRITON_INTERPRET"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import warnings

import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear
from src.ops.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.triton import monarch_kernel

seq_len = 512
nblocks = 4
in_dim = 1024
out_dim = 1024
blk_r = 16

warmup_iter = 5
num_bench_iter = 200


def main(args):
    torch.cuda.manual_seed_all(0)
    x = torch.randn(seq_len, in_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    x_copy = x.clone().detach().requires_grad_(True)
    # Ongoing issues with Triton bmm
    # https://github.com/triton-lang/triton/issues/5424
    # https://github.com/triton-lang/triton/issues/5211
    # https://github.com/triton-lang/triton/pull/5285
    monarch = MonarchLinear(
        in_dim, out_dim, nblocks, blk_r=blk_r, bias=False, as_adapter=False, use_triton=False, dtype=torch.float16
    ).cuda()
    if os.environ.get("TRITON_INTERPRET", None) == "1":
        warnings.warn(
            "Running in TRITON_INTERPRET mode. This will make kernels super slow and is only for debugging purposes."
        )
        if x.dtype == torch.bfloat16:
            raise NotImplementedError("TRITON_INTERPRET mode does not support bfloat16")

    if args.test:
        out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        blkdiag1_copy = monarch.blkdiag1.clone().detach().requires_grad_(True)
        blkdiag2_copy = monarch.blkdiag2.clone().detach().requires_grad_(True)
        out2_triton, out1_triton = monarch_kernel(x_copy, blkdiag1_copy, blkdiag2_copy, True)
        # print(out1_triton)

        # Forward tests
        assert_close(out1_torch, out1_triton, rtol=1e-3, atol=1e-3)
        assert_close(out2_torch, out2_triton, rtol=1e-3, atol=1e-3)
        # Backward
        # out2_torch.sum().backward()
        # out2_triton.sum().backward()
        # assert_close(monarch.blkdiag1.grad, blkdiag1_copy.grad, rtol=1e-3, atol=1e-3)
        # assert_close(monarch.blkdiag2.grad, blkdiag2_copy.grad, rtol=1e-3, atol=1e-3)
        # assert_close(x.grad, x_copy.grad, rtol=1e-3, atol=1e-3)

        print("Precision tests passed!")
        print()

        del out1_torch, out1_triton, out2_torch, out2_triton, x_copy
        # test memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.max_memory_allocated() / 1024**2
        out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        mem = torch.cuda.max_memory_allocated() / 1024**2 - mem_before
        print(f"PyTorch peak (activation) memory usage: {mem} MB")
        del out1_torch, out2_torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.max_memory_allocated() / 1024**2
        out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)
        mem = torch.cuda.max_memory_allocated() / 1024**2 - mem_before
        print(f"Triton peak (activation) memory usage: {mem} MB")
        del out1_triton, out2_triton
        torch.cuda.empty_cache()

    if args.benchmark:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_iter):
            out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)

        start_event.record()
        for _ in range(num_bench_iter):
            out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        end_event.record()
        end_event.synchronize()
        print(f"PyTorch time: {start_event.elapsed_time(end_event)} ms")

        # Warmup
        for _ in range(10):
            out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)

        start_event.record()
        for _ in range(num_bench_iter):
            out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)
        end_event.record()
        end_event.synchronize()
        print(f"Triton time: {start_event.elapsed_time(end_event)} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=True, type=eval, help="Run test")
    parser.add_argument("-b", "--benchmark", default=True, type=eval, help="Run speed benchmark")
    args = parser.parse_args()
    main(args)
