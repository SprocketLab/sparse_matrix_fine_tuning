import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TRITON_INTERPRET"] = "1"

import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear

seq_len = 64
nblocks = 4
in_dim = 128
out_dim = 64
blk_r = 16
x = torch.randn(seq_len, in_dim, device="cuda", dtype=torch.float16)  # TODO: Triton has bug in loading bf16?
monarch = MonarchLinear(
    in_dim, out_dim, nblocks, blk_r=blk_r, bias=False, as_adapter=False, use_triton=False, dtype=torch.bfloat16
).cuda()

y1 = monarch(x)
y2 = monarch(x, use_triton=True)
print(y1)
print(y2)
assert_close(y1, y2, rtol=1e-4, atol=1e-4)
print("Test passed!")
