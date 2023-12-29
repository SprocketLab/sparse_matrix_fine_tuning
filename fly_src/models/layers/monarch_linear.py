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

hooked = False
check_freq = 500
check_step = 0

def hook_fn(module, grad_input, grad_output):
    # print(f"{module}'s output's grad (dy) is {grad_output}")
    global check_freq, check_step
    if check_step % check_freq == 0:
        print(f"{module} has dW {grad_input[1]} and scaler value {module.scaler}")
    check_step += 1
    
    
def factor_balance(mid_blksz, out_blksz):
    total = mid_blksz * out_blksz

# @Wenxuan
# Use a diagonal matrix or a scaler to scale output of monarch factors
class Scaler(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.scaler = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x = self.scaler * x
        x = F.layer_norm(x, x.shape[1:])
        # layernorm to avoid vanishing gradient
        return x

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
        if self.peft and rank > 1:
            raise NotImplementedError("Adapters with rank > 1 can't be merged with dense weights due to dim mismatch")
        
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
        self.scaler = Scaler(self.out_features)
        
        self.reset_parameters()
        
        if weights is not None:
            self.set_weights_from_dense_init(weights, rank)

        self.to(device)
        
        # Hook only one layer to debug
        global hooked
        if not hooked:
            self.hook = self.scaler.register_full_backward_hook(hook_fn)
            hooked = True
            
        logger.info(f"Linear class {self.__class__}: saving={self.saving}")


    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        # if self.peft:
        #     monarch_factors = [self.blkdiag1] # set the second factor to 0 to init at the pretrained point
        # else:
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
        super().train(mode)
        if mode:
            if self.peft and self.merged:
                # # split out monarch for separate training
                self.dense.data -= blockdiag_butterfly_multiply(
                    torch.eye(self.in_features, device=self.device), self.blkdiag1, self.blkdiag2
                )
                # re-register monarch weights
                if isinstance(self.blkdiag1, torch.Tensor):
                    self.blkdiag1 = nn.Parameter(self.blkdiag1)
                    self.blkdiag2 = nn.Parameter(self.blkdiag2)

                self.merged = False
                self.dense.requires_grad_(False) # freeze dense, train monarch adapter
        else:
            if self.peft and not self.merged:
                # Merge the weights and mark it
                self.dense.data += blockdiag_butterfly_multiply(
                    torch.eye(self.in_features, device=self.device), self.blkdiag1, self.blkdiag2
                )
                # unregister the monarch weights
                # self.blkdiag1 = self.blkdiag1.data
                # self.blkdiag2 = self.blkdiag2.data
                data1 = self.blkdiag1.data
                data2 = self.blkdiag2.data
                del self.blkdiag1, self.blkdiag2
                self.blkdiag1 = data1
                self.blkdiag2 = data2
                self.merged = True

    def forward(self, x):
        if self.peft:
            if self.merged:
                # inference
                x = F.linear(x, self.dense)
            else:
                # training
                x = self.monarch_forward(x) + F.linear(x, self.dense)
        else:
            x = self.monarch_forward(x)
        return self.scaler(x) + self.bias


    # Override magic methods
    def __repr__(self):
        weight_shape = self.nblocks = {self.blkdiag1.shape[0]} if self.blkdiag1 is not None else self.dense.shape
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_shape={weight_shape}, requires_grad={list(self.parameters())[0].requires_grad})"
        )
    
    # TODO: make it actually merge when saving 
    # def __getstate__(self):
    #     """
    #     Override to remove the dense weights from state dict
    #     """
    #     state = super().__getstate__()
    #     if self.peft:
    #         state["dense"] = None
    #     return state