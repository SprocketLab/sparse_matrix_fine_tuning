import os
import sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear
from src.ops.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.triton import monarch_kernel

seq_len = 16
nblocks = 4
in_dim = 16
out_dim = 16 # setting to 64 cause seg fault using TRITON_INTERPRET=1 and precision error without the flag
blk_r = 16

warmup_iter = 5
num_bench_iter = 100

def main(args):
    torch.cuda.manual_seed_all(0)
    x = torch.randn(seq_len, in_dim, device="cuda", dtype=torch.float16) # seems like bf16 has precision err

    # Ongoing issues with Triton bmm
    # https://github.com/triton-lang/triton/issues/5424
    # https://github.com/triton-lang/triton/issues/5211
    # https://github.com/triton-lang/triton/pull/5285
    monarch = MonarchLinear(
        in_dim, out_dim, nblocks, blk_r=blk_r, bias=False, as_adapter=False, use_triton=False, dtype=torch.float16
    ).cuda()
    if args.test:

        out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
        # print(out1_torch)
        out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)

        # print(out1_triton)
        assert_close(out1_torch, out1_triton, rtol=1e-3, atol=1e-3)
        assert_close(out2_torch, out2_triton, rtol=1e-3, atol=1e-3)
        print("Test passed!")
        
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
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run speed benchmark")
    args = parser.parse_args()
    main(args)
