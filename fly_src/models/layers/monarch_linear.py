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
check_freq = 600
check_step = 0

def backward_hook(module, grad_input, grad_output):
    global check_freq, check_step
    if check_step % check_freq == 0:
        print(f"{module} has mean dX {grad_input[0].mean()} and scaler value {module.scaler}")
    check_step += 1

def grad_hook(grad):
    global check_freq, check_step
    if check_step % check_freq == 0:
        print(f"Scaler has  dW {grad}")  
    check_step += 1
    
def factor_balance(mid_blksz, out_blksz):
    total = mid_blksz * out_blksz

# @Wenxuan
class Scaler(nn.Module):
    """
        Scale output of monarch factors
    """
    def __init__(self, out_features, scaler_type="scaler"):
        super().__init__()
        assert scaler_type in ["scaler", "diag"]
        self.scaler_type = scaler_type
        
        if scaler_type == "scaler":
            self.scaler = nn.Parameter((torch.zeros(1)))
        else:
            self.scaler = nn.Parameter((torch.zeros(out_features)))
            
        # hook only module
        global hooked
        if not hooked:
            self.hook = self.scaler.register_hook(grad_hook)
            hooked = True
            
    def forward(self, x):
        if self.scaler_type == "scaler":
            x = x * self.scaler
        else:
            x = x @ torch.diag(self.scaler)
        # x = F.layer_norm(x, x.shape[1:])
        return x

""" @Wenxuan """
class MonarchLinear(StructuredLinear):
    """
    bmm with two monarch factors
    """
    def __init__(
        self,
        *args,
        peft_config,
        nblocks=4,
        weights=None,
        rank=1,
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
            use_scaler (bool, optional): whether to scale the output of monarch factors
        """
        super().__init__(*args, **kwargs)

        self.nblocks = nblocks
        self.in_blksz = int(math.ceil(self.in_features / nblocks))
        self.mid_blksz = self.nblocks * rank
        self.out_blksz = int(math.ceil(self.out_features / nblocks)) * rank
        
        # Get configs
        self.device = device
        self.peft_config = peft_config
        self.peft = peft_config["use_peft"]
        self.use_scaler = peft_config.get("use_scaler", False)
        self.rank = rank
        self.lora_style_init = peft_config.get("lora_style_init", False)
        self.scaler_type = peft_config.get("scaler_type", "scaler")
        assert self.scaler_type in ["scaler", "diag"]
        
        if self.peft and rank > 1:
            raise NotImplementedError("Adapters with rank > 1 can't be merged with dense weights due to dim mismatch")
        
        self.merged = False
        assert self.rank <= min(
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
            if self.peft:
                self.dense = nn.Parameter(weights, requires_grad=False)
            else:
                self.set_weights_from_dense_init(weights, rank)

        self.to(device)
        
        # Set a scaling matrix
        if self.use_scaler: 
            if self.lora_style_init:
                raise ValueError("LoRA init already zeroed out; no need for scaler")
            self.scaler = Scaler(self.out_features, self.scaler_type)
        else:
            self.scaler = nn.Identity()
        self.scaler.to(self.device)
        
        logger.info(f"Linear class {self.__class__}: saving={self.saving}")


    def reset_parameters(self) -> None:
        monarch_factors = [self.blkdiag1]
        if self.use_scaler:
            monarch_factors.append(self.blkdiag2) # zero init the scaler only
            
        if self.lora_style_init:
            lora_rank = 4
            lora_A = torch.zeros(lora_rank, self.in_features)
            self.set_weights_from_dense_init(lora_A, rank=self.rank)
            # zero out 2nd monarch factor to start training from checkpoint
            self.blkdiag2.data.zero_()
        else:
            for blkdiag in monarch_factors:
                # init.kaiming_uniform_(blkdiag, a=math.sqrt(5)) # sqrt(5) should cancel "gain" out and give uniform(-1 / std, 1 / std)
                
                ## Mimic init.kaiming_uniform but only on each block: p of (k, q, p) instead of q * p
                ## https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
                fan_in = blkdiag.shape[-1]
                gain = init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5)) 
                std = gain / math.sqrt(fan_in)
                bound = (
                    math.sqrt(3.0) * std
                )  
                ## Calculate uniform bounds from standard deviation
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
                # split out monarch for separate training
                # (out, in) + (in, out).T
                self.dense.data -= blockdiag_butterfly_multiply(
                    torch.eye(self.in_features, device=self.device), self.blkdiag1, self.blkdiag2
                ).T
                self.merged = False
            self.dense.requires_grad_(False) # freeze dense, train monarch adapter
            
        else:
            if self.peft and not self.merged:
                # Merge the weights and mark it
                self.dense.data += blockdiag_butterfly_multiply(
                    torch.eye(self.in_features, device=self.device), self.blkdiag1, self.blkdiag2
                ).T
                self.merged = True


    def forward(self, x):
        if self.peft:
            if self.merged:
                # inference
                x = F.linear(x, self.dense)
            else:
                # training
                x = self.scaler(self.monarch_forward(x)) + F.linear(x, self.dense)
        else:
            x = self.scaler(self.monarch_forward(x))
        return x + self.bias


    # Override magic methods
    def __repr__(self):
        weight_shape = self.nblocks = {self.blkdiag1.shape[0]} if self.blkdiag1 is not None else self.dense.shape
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_shape={weight_shape}, requires_grad={list(self.parameters())[0].requires_grad})"
        )
    