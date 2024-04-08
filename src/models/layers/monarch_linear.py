import math

import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange
import torch.nn.functional as F
from src.models.layers.structured_linear import StructuredLinear
from src.models.layers.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply, single_monarch_mult
from src.utils.utils import get_logger

# NOTE converting weights to monarch matrices
from src.ops.blockdiag_butterfly_projection import (
    blockdiag_butterfly_project,
)  # square weights, rank 1
from src.ops.blockdiag_butterfly_einsum import (
    blockdiag_butterfly_project_einsum_rank,  # for rectangular, custom rank
    blockdiag_butterfly_project_einsum_simple,  # for rectangular, rank 1
)

logger = get_logger()

hooked = False
hooked_module = None
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
        print(f"Scaler has  dW {grad} and value {hooked_module.scaler}. ")  
    check_step += 1
    
def factor_balance(mid_blksz, out_blksz):
    total = mid_blksz * out_blksz

# @Wenxuan
class Scaler(nn.Module):
    """
        Scale output of monarch factors
    """
    def __init__(self, out_features, scaler_type="scaler", affine=False, layernorm=False):
        super().__init__()
        assert scaler_type in ["scaler", "diag"]
        self.scaler_type = scaler_type
        
        if scaler_type == "scaler":
            self.scaler = nn.Parameter((torch.zeros(1)))
        else:
            self.scaler = nn.Parameter((torch.zeros(out_features)))
        self.norm = nn.LayerNorm(out_features, elementwise_affine=affine)
        
        # hook only module
        global hooked, hooked_module
        if not hooked:
            hooked_module = self
            self.hook = self.scaler.register_hook(grad_hook)
            hooked = True
            
    def forward(self, x):
        if self.scaler_type == "scaler":
            x = x * self.scaler
        else:
            x = x @ torch.diag(self.scaler)
        x = self.norm(x)
        return x

""" @Wenxuan """
_DEFAULT_CONFIG = {
    "nblocks": 4,
    "blk_r": 4,
    "blk_sz": None,
    "square": False,
    "adapter": True,
}
class MonarchLinear(StructuredLinear):
    """
    bmm with two monarch factors
    """
    def __init__(
        self,
        in_features,
        out_features,
        peft_config: dict = _DEFAULT_CONFIG,
        nblocks: int = 4,
        weights: torch.Tensor = None,
        device="cuda",
        *args,
        **kwargs,
    ):
        """
        Args:
            nblocks (int, optional): Number of blocks in block-diag monarch factor. More blocks -> less precision loss in SVD
            weights (torch.Tensor, optional): The dense weight matrix for projection. If none will init with Kaiming
            blk_r (int, optional): The per block rank (output dim). Used also in SVD projection and param definition
            blk_sz (int, optional): Size of each block. If None, will be calculated from in_features
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.nblocks = nblocks
        self.blk_r = peft_config["blk_r"] if "blk_r" not in kwargs  else kwargs["blk_r"]
        self.blk_sz = peft_config["blk_sz"] if "blk_sz" not in kwargs else kwargs["blk_sz"]
        
        if self.blk_sz is None:
            self.blk_sz = int(math.ceil(self.in_features / nblocks))
        self.in_blksz = self.blk_sz
        self.mid_blksz = self.blk_r
        # Use square blocks if testing block size trade-offs
        if peft_config["square"]:
            self.mid_blksz = self.in_blksz
            
        # Throw away blocks that are fully padded 
        if self.nblocks * self.in_blksz > self.in_features:
            self.nblocks = (self.in_features + self.in_blksz - 1) // self.in_blksz 
        elif self.nblocks * self.in_blksz < self.in_features:
            self.nblocks = (self.in_features + self.in_blksz - 1) // self.in_blksz
                
        align_factor = int(math.ceil(self.out_features / self.in_features)) # Useful for the blocks in the two monarch factors to exactly match  
        self.out_blksz = self.in_blksz * align_factor
        
        # Get peft configs
        self.device = device
        self.peft_config = peft_config
        self.use_adapter = peft_config["adapter"]
        self.use_scaler = peft_config.get("scaler", False)
        self.lora_style_init = peft_config.get("lora_style_init", False)
        self.scaler_type = peft_config.get("scaler_type", "scaler")
        self.use_mult_factor = peft_config.get("use_mult_factor", False)
        self.merged = False
        self.use_scaler = self.use_scaler or self.use_mult_factor
        
        assert self.scaler_type in ["scaler", "diag"]
        assert self.blk_r <= min(
            self.in_blksz, self.out_blksz
        ), "rank must be smaller than the smaller block size"
        
        # Init block-diagonal monarch factors 
        self.blkdiag1 = nn.Parameter(
                torch.zeros(self.nblocks, self.mid_blksz, self.in_blksz, device=self.device) # (nblocks, r * nblocks , in_features / nblocks)
        )  
        self.blkdiag2 = nn.Parameter(
                torch.zeros(self.nblocks, self.out_blksz, self.mid_blksz, device=self.device) # (nblocks, out_features / nblocks, r * nblocks)
        )
        
        # init a batch of identity matrices as a multiplicative factor
        # X @ W @ M_mult + X @ M1 @ M2 * scaler
        # (nblocks, out_features / nblocks, in_features / nblocks)
        if self.use_mult_factor:
            self.blkdiag_mult = nn.Parameter(
                torch.eye(self.out_blksz, self.in_blksz, device=self.device).repeat(nblocks, 1, 1)
            )
        
        # initialize frozen dense weights
        self.reset_parameters()
        if weights is not None:
            if self.use_adapter:
                self.dense = nn.Parameter(weights, requires_grad=False)
            else:
                self.set_weights_from_dense_init(weights, self.blk_r)
        self.to(device)
        
        # Initialize scaling (vector or a scaler)
        if self.use_scaler: 
            if self.lora_style_init:
                raise ValueError("LoRA init already zeroed out; no need for scaler")
            layernorm = peft_config.get("layernorm", False)
            affine = peft_config.get("affine", False)
            self.scaler = Scaler(self.out_features, self.scaler_type, affine, layernorm)
        else:
            self.scaler = nn.Identity()
        self.scaler.to(self.device)
        logger.info(f"Linear class {self.__class__}: saving={self.saving}")


    def reset_parameters(self) -> None:
        """
        Initialize block-diagonal weights and biases
        """
        monarch_factors = [self.blkdiag1]
        if self.use_scaler:
            monarch_factors.append(self.blkdiag2) # zero init the scaler only
            
        if self.lora_style_init:
            lora_rank = 4
            lora_A = torch.zeros(lora_rank, self.in_features)
            self.set_weights_from_dense_init(lora_A, rank=self.blk_r)
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


    def monarch_forward(self, x):
        """
        Forward using monarch factors only
        """
        output = blockdiag_butterfly_multiply(
            self.preprocess(x), self.blkdiag1, self.blkdiag2
        )
        return self.scaler(self.postprocess(output))


    def set_weights_from_dense_init(self, w: torch.Tensor, rank=1):
        """
        Args:
            w (torch.Tensor): Dense weight matrix to be projected using SVD
            rank (int, optional): SVD rank
        """
        assert w.ndim == 2, "w must be a 2D weight matrix"
        # project to monarch matrix
        # blkdiag1, blkdiag2 = blockdiag_butterfly_project(w)
        blkdiag1, blkdiag2 = blockdiag_butterfly_project_einsum_rank(
            w, self.nblocks, self.nblocks, rank
        )
        assert blkdiag1.shape == self.blkdiag1.shape and blkdiag2.shape == self.blkdiag2.shape, \
            "Projected monarch shapes mismatch original shapes. Check you dense weight shape!"
        self.blkdiag1 = nn.Parameter(blkdiag1)
        self.blkdiag2 = nn.Parameter(blkdiag2)
        
        
    def train(self, mode: bool = True):
        """
        Override for freezing and merging weights
        """
        super().train(mode)
        if mode:
            if self.use_adapter:
                if self.merged:
                    # split out monarch for separate training
                    # (out, in) - (in, out).T
                    merged_weights = self.monarch_forward(torch.eye(self.in_features, device=self.device)).T
                    self.dense.data -= merged_weights
                    self.merged = False
                self.dense.requires_grad_(False) # freeze dense, train monarch adapter
                self.bias.requires_grad_(False)
        else:
            if self.use_adapter and not self.merged:
                # Merge the adapter weights and mark it
                merged_weights = self.monarch_forward(torch.eye(self.in_features, device=self.device)).T
                self.dense.data += merged_weights
                self.merged = True


    def forward(self, x):
        if self.use_adapter:
            out = F.linear(x, self.dense) if hasattr(self, "dense") else x
            if self.use_mult_factor:
                out = single_monarch_mult(out, self.blkdiag_mult)
                
            if not self.merged:                
                # training with adapter
                x = out + self.monarch_forward(x)
            else:
                x = out
        else:
            # Dense already projected to monarch
            x = self.monarch_forward(x)
        
        return x + self.bias if getattr(self, "bias", None) is not None else x


    # Override magic methods
    def __repr__(self):
        # weight_shape = self.nblocks = {self.blkdiag1.shape[0]} if self.blkdiag1 is not None else self.dense.shape
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, "
            f"nblocks={self.nblocks}, requires_grad={list(self.parameters())[0].requires_grad})"
        )
        
    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.nblocks * self.in_blksz:
            x = F.pad(x, (0, self.nblocks * self.in_blksz - in_features))
        return x
    
    @property
    def saving(self):
        return (self.blkdiag1.numel() + self.blkdiag2.numel()) / (
            self.in_features * self.out_features
        )