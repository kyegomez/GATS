import torch
from torch import nn, Tensor
from zeta.nn import (
    img_to_text,
    video_to_text,
    audio_to_text,
)
from local_attention import LocalAttention



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
        seqlen: int = 1028,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.seqlen = seqlen

        inner_dim = dim_head * heads

        self.local_attn = LocalAttention(
            window_size,
            causal,
            look_backward,
            look_forward,
            dropout,
            dim=dim,
            autopad=True,
            *args,
            **kwargs,
        )

    def forward(
        self,
        text: Tensor, # 3d Tensor - (B, T, S)
        img: Tensor = None, # 4d Tensor - (B, C, H, W)
        audio: Tensor = None, # 3d Tensor - (B, T)
        video: Tensor = None, # 5d Tensor - (B, T, C, H, W)
        action: Tensor = None, # 7D Tensor - 
        mask: Tensor = None,
    ):
        img = img_to_text(img, self.seqlen, self.dim, True)
        audio = audio_to_text(audio, self.seqlen, self.dim, True)
        video = video_to_text(video, self.seqlen, self.dim, True)
        
        x = torch.cat((text, img, audio, video))
        x = self.local_attn(text, audio, video)
        
        print(x.shape)
        return x

