import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# os.environ["TRITON_INTERPRET"] = "1"
import argparse
import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear
from src.ops.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.triton import monarch_kernel
import time

seq_len = 64
nblocks = 4
in_dim = 128
out_dim = 64
blk_r = 16
num_bench_iter = 100

def main(args):
    x = torch.randn(seq_len, in_dim, device="cuda", dtype=torch.float16)
    # NOTE: It looks like Triton has some bugs with 3D matmul for now, so this 
    # implementation might not work as expected.
    # See issues
    # https://github.com/triton-lang/triton/issues/5424
    # https://github.com/triton-lang/triton/issues/5211
    # https://github.com/triton-lang/triton/pull/5285
    monarch = MonarchLinear(
        in_dim, out_dim, nblocks, blk_r=blk_r, bias=False, as_adapter=False, use_triton=False, dtype=torch.float16
    ).cuda()
    if args.test:

        breakpoint()
        out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        print(out1_torch)
        out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)

        print(out1_triton)
        assert_close(out1_torch, out1_triton, rtol=1e-5, atol=1e-5)
        assert_close(out2_torch, out2_triton, rtol=1e-4, atol=1e-4)
        print("Test passed!")
        
    if args.benchmark:
        t1 = time.time()
        for _ in range(num_bench_iter):
            out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        torch.cuda.synchronize()
        t2 = time.time()
        
        # Warmup
        for _ in range(10):
            out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)
            
        t3 = time.time()    
        for _ in range(num_bench_iter):
            out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)
        torch.cuda.synchronize()
        t4 = time.time()
        print(f"PyTorch time: {t2-t1}")
        print(f"Triton time: {t4-t3}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=True, type=eval, help="Run test")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run speed benchmark")
    args = parser.parse_args()
    main(args)
