import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TRITON_INTERPRET"] = "1"

import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear
from src.ops.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.ops.triton import monarch_kernel

seq_len = 64
nblocks = 4
in_dim = 128
out_dim = 64
blk_r = 16
x = torch.randn(seq_len, in_dim, device="cuda", dtype=torch.float16)
monarch = MonarchLinear(
    in_dim, out_dim, nblocks, blk_r=blk_r, bias=False, as_adapter=False, use_triton=False, dtype=torch.float16
).cuda()

# out2_triton = monarch(x)
# out2_torch = monarch(x, use_triton=True)
out2_torch, out1_torch = blockdiag_butterfly_multiply(x, monarch.blkdiag1, monarch.blkdiag2, True)
out2_triton, out1_triton = monarch_kernel(x, monarch.blkdiag1, monarch.blkdiag2, True)

print(out2_torch)
print(out2_triton)
assert_close(out1_torch, out1_triton, rtol=1e-4, atol=1e-4)
assert_close(out2_torch, out2_triton, rtol=1e-4, atol=1e-4)
print("Test passed!")
