import torch
from torch import nn, Tensor
from einops import rearrange, reduce
from zeta.nn import LocalAttention, Attention
from classifier_free_guidance_pytorch import (
    classifier_free_guidance_class_decorator,
)
from zeta import exists


@classifier_free_guidance_class_decorator
class GATSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        window_size: int = 512,
        causal: bool = True,
        look_backward: int = 1,
        look_forward: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

        inner_dim = dim_head * heads

        self.local_attn = LocalAttention(
            window_size,
            causal,
            look_backward,
            look_forward,
            dropout,
            dim=dim,
            *args,
            **kwargs,
        )
    

    def forward(
        self,
        text: Tensor,
        img: Tensor = None,
        audio: Tensor = None,
        Video: Tensor = None,
    ):
        pass