import math

import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange
import torch.nn.functional as F
from fly_src.models.layers.structured_linear import StructuredLinear
from fly_src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from fly_src.utils.utils import get_logger

# NOTE converting weights to monarch matrices
from fly_src.ops.blockdiag_butterfly_projection import (
    blockdiag_butterfly_project,
)  # square weights, rank 1
from fly_src.ops.blockdiag_butterfly_einsum import (
    blockdiag_butterfly_project_einsum_rank,  # for rectangular, custom rank
    blockdiag_butterfly_project_einsum_simple,  # for rectangular, rank 1
)

logger = get_logger()

def factor_balance(mid_blksz, out_blksz):
    total = mid_blksz * out_blksz

# @Wenxuan
class MonarchLinear(StructuredLinear):
    """
    The original class supports Dense-init training only. Modified it for dense to sparse training.
    """

    def __init__(
        self,
        *args,
        nblocks=4,
        weights=None,
        rank=1,
        peft=False,
        device="cuda",
        **kwargs,
    ):
        """
        Args:
            nblocks (int, optional): Number of blocks in block-diag monarch factor. More blocks -> less precision loss in SVD
            weights (torch.Tensor, optional): The dense weight matrix for projection. If none will init with Kaiming
            rank (int, optional): SVD rank for decomposing each block
            peft (bool, optional): whether to use PEFT(freeze dense, train task-specific monarch matrices and fuse)
                or FT (project dense to and train monarch matrices only).
        """
        super().__init__(*args, **kwargs)

        self.nblocks = nblocks
        self.in_blksz = int(math.ceil(self.in_features / nblocks))
        self.mid_blksz = self.nblocks * rank
        self.out_blksz = int(math.ceil(self.out_features / nblocks)) * rank
        

        # Get actual input/output features without permutation

        self.device = device
        self.peft = peft
        self.merged = False
        assert rank <= min(
            self.in_blksz, self.out_blksz
        ), "rank must be smaller than the smaller block size"
        
        # Init weights
        self.blkdiag1 = nn.Parameter(
                torch.zeros(nblocks, self.mid_blksz, self.in_blksz) # (nblocks, r * nblocks, i)
        )  
        self.blkdiag2 = nn.Parameter(
                torch.zeros(nblocks, self.out_blksz, self.mid_blksz) # (nblocks, l, nblocks * r)
        )
        self.reset_parameters()
        
        if weights is not None:
            self.set_weights_from_dense_init(weights, rank)

        self.to(device)
            
        logger.info(f"Linear class {self.__class__}: saving={self.saving}")


    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        if self.peft:
            monarch_factors = [self.blkdiag1] # set the second factor to 0 to init at the pretrained point
        else:
            monarch_factors = [self.blkdiag1, self.blkdiag2]
            
        for blkdiag in monarch_factors:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = (
                math.sqrt(3.0) * std
            )  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()


    @property
    def saving(self):
        return (self.blkdiag1.numel() + self.blkdiag2.numel()) / (
            self.in_features * self.out_features
        )


    def monarch_forward(self, x):
        output = blockdiag_butterfly_multiply(
            self.preprocess(x), self.blkdiag1, self.blkdiag2
        )
        return self.postprocess(output)


    def set_weights_from_dense_init(self, w: torch.Tensor, rank=1):
        """
        Args:
            w (torch.Tensor): Dense weight matrix to be projected using SVD
            rank (int, optional): SVD rank
        """
        assert w.ndim == 2, "w must be a 2D weight matrix"
        is_square = w.shape[0] == w.shape[1]
        if self.peft:
            self.dense = nn.Parameter(w)
        else:
            # project to monarch matrix
            # blkdiag1, blkdiag2 = blockdiag_butterfly_project(w)
            blkdiag1, blkdiag2 = blockdiag_butterfly_project_einsum_rank(
                w, self.nblocks, self.nblocks, rank
            )
            self.blkdiag1 = nn.Parameter(blkdiag1)
            self.blkdiag2 = nn.Parameter(blkdiag2)

    def train(self, mode: bool = True):
        """
        Override for freezing and merging weights
        """
        
        if mode:
            if self.peft:
                # # split out monarch for separate training
                # self.dense.data -= blockdiag_butterfly_multiply(
                #     torch.eye(self.in_features), self.blkdiag1, self.blkdiag2
                # )
                # self.merged = False
                self.dense.requires_grad_(False) # freeze dense
        else:
            if self.peft and not self.merged:
                # Merge the weights and mark it
                self.dense.data += blockdiag_butterfly_multiply(
                    torch.eye(self.in_features, device=self.device), self.blkdiag1, self.blkdiag2
                )
                self.blkdiag1 = self.blkdiag2 = None # save storage
                self.merged = True


    def forward(self, x):
        if self.peft:
            if self.merged:
                return F.linear(x, self.dense, self.bias)
            else:
                return self.monarch_forward(x) + F.linear(x, self.dense, self.bias)
        else:
            return self.monarch_forward(x)


    def __repr__(self):
        weight_shape = nblocks={self.blkdiag1.shape[0]} if self.blkdiag1 is not None else self.dense.shape
        return (
            f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
            f"{weight_shape}, requires_grad={list(self.parameters())[0].requires_grad})"
        )