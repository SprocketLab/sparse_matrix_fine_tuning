import os

os.environ["TRITON_INTERPRET"] = "1"
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.testing import assert_close

from src.layers.monarch_linear import MonarchLinear

seq_len = 1024
nblocks = 4
in_dim = 1024
out_dim = 1024
x = torch.randn(seq_len, in_dim, device="cuda")
monarch = MonarchLinear(in_dim, out_dim, nblocks, bias=False, as_adapter=False, use_triton=False).cuda()

y1 = monarch(x)
y2 = monarch(x, use_triton=True)
print(y1)
print(y2)
assert_close(y1, y2, rtol=1e-4, atol=1e-4)
print("Test passed!")
