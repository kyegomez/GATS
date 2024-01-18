import torch
from torch import nn, Tensor
from zeta.nn import (
    img_to_text,
    video_to_text,
    audio_to_text,
    Attention,
    FeedForward,
)
from local_attention import LocalAttention
from einops import rearrange

class GATSBlock(nn.Module):
    """
    GATSBlock is a module that represents a single block of the GATS (Graph Attention Time Series) model.

    Args:
        dim (int): The input dimension of the block.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        window_size (int, optional): The window size for local attention. Defaults to 512.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        look_backward (int, optional): The number of tokens to look backward in local attention. Defaults to 1.
        look_forward (int, optional): The number of tokens to look forward in local attention. Defaults to 0.
        seqlen (int, optional): The maximum sequence length. Defaults to 1028.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (nn.Dropout): The dropout layer.
        window_size (int): The window size for local attention.
        seqlen (int): The maximum sequence length.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        local_attn (LocalAttention): The local attention module.
        attn (Attention): The attention module.
        ffn (FeedForward): The feed-forward network module.

    """

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
        ff_mult: int = 4,
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
        self.ff_mult = ff_mult

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
        
        self.attn = Attention(
            dim,
            dim_head,
            heads,
            causal,
            flash=False,
            dropout=dropout,
            qk_norm=True,   
        )
        
        self.ffn = FeedForward(
            dim,
            dim,
            ff_mult,
            post_act_ln=True
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
        """
        Forward pass of the GATSBlock.

        Args:
            text (Tensor): The input text tensor of shape (B, T, S).
            img (Tensor, optional): The input image tensor of shape (B, C, H, W). Defaults to None.
            audio (Tensor, optional): The input audio tensor of shape (B, T). Defaults to None.
            video (Tensor, optional): The input video tensor of shape (B, T, C, H, W). Defaults to None.
            action (Tensor, optional): The input action tensor. Defaults to None.
            mask (Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The output text, image, and audio tensors.

        """
        img_b, img_c, h, w = img.shape
        img = img_to_text(img, self.seqlen, self.dim, True)
        audio = audio_to_text(audio, self.seqlen, self.dim, True)
        video = video_to_text(video, self.seqlen, self.dim, True)
        
        x = torch.cat((text, img, audio, video))
        x = self.local_attn(text, audio, video)
        
        # Attention
        x, _ = self.attn(x)
        x = x + x
        
        # FFn with + residual
        x = self.ffn(x) + x
        
        # Scatter back to modalities
        text = x
        img = rearrange(x, "B (H W) D -> B D H W", h=h, w=w)
        audio = rearrange(x, "B T D -> B D")
        # video = rearrange(x, "B S D -> ")
        
        return text, img, audio #, video
        # return x

