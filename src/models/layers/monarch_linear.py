import math

import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange

from src.models.layers.structured_linear import StructuredLinear
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from src.utils.utils import get_logger

# NOTE converting weights to monarch matrices
from src.ops.blockdiag_butterfly_projection import blockdiag_butterfly_project # square weights, rank 1
from src.ops.blockdiag_butterfly_einsum import (
    blockdiag_butterfly_project_einsum_rank, # for rectangular, custom rank 
    blockdiag_butterfly_project_einsum_simple # for rectangular, rank 1
    )
logger = get_logger()

# @Wenxuan
class MonarchLinear(StructuredLinear):
    """
    The original class supports Dense-init training only. Modified it for dense to sparse training. 
    """
    def __init__(self, *args, nblocks=4, weights: torch.Tensor=None, rank=1, device="cuda", **kwargs):
        """
        Args:
            nblocks (int, optional): _description_. Defaults to 4.
            weights (torch.Tensor, optional): dense weight matrix for projection. If none will init with Kaiming
            rank (int, optional): _description_. Defaults to 1.
            device (str, optional): _description_. Defaults to "cuda".
        """
        super().__init__(*args, **kwargs)
        
        in_blksz = int(math.ceil(self.in_features / nblocks))
        mid_blksz = in_blksz * rank
        out_blksz = int(math.ceil(self.out_features / nblocks)) * rank
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.device = device
        assert rank <= min(in_blksz, out_blksz), "rank must be smaller than the smaller block size"
        
        if weights is not None:
            self.set_weights_from_dense_init(weights, rank)
        else:
            if self.in_features_extended < self.out_features_extended:
                self.blkdiag1 = nn.Parameter(torch.empty(nblocks, mid_blksz, in_blksz))
                self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, mid_blksz)) 
            else:
                self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
                self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))    
            self.reset_parameters()
        
        self.to(device)
        logger.info(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5)) 
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)
    
    def set_weights_from_dense_init(self, w: torch.Tensor, rank = 1):
        assert w.ndim == 2, "w must be a 2D weight matrix"
        is_square = (w.shape[0] == w.shape[1])
        # project to monarch matrix
        if is_square:
            blkdiag1, blkdiag2 = blockdiag_butterfly_project(w)
        else:
            blkdiag1, blkdiag2 = blockdiag_butterfly_project_einsum_rank(w, rank=rank)
        self.blkdiag1 = nn.Parameter(blkdiag1)
        self.blkdiag2 = nn.Parameter(blkdiag2)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_features}, {self.out_features}, '
                f'nblocks={self.blkdiag1.shape[0]}, requires_grad={list(self.parameters())[0].requires_grad})')
        