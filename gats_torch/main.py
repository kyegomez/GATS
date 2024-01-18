import torch
from torch import nn, Tensor
from zeta.nn import (
    # LocalAttention,
    img_to_text,
    video_to_text,
    audio_to_text,
)
from local_attention import LocalAttention




# @classifier_free_guidance_class_decorator
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

        dim_head * heads

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
        text: Tensor,
        img: Tensor = None,
        audio: Tensor = None,
        video: Tensor = None,
        action: Tensor = None,
        mask: Tensor = None,
    ):
        img = img_to_text(img, self.seqlen, self.dim, True)
        audio = audio_to_text(audio, self.seqlen, self.dim, True)
        video = video_to_text(video, self.seqlen, self.dim, True)
        
        x = torch.cat((text, img, audio, video))
        x = self.local_attn(text, audio, video)
        
        print(x.shape)
        return x

model = GATSBlock(
    dim=512,
    heads=8,
    dim_head=64,
    dropout=0.1,
    window_size=512,
    causal=True,
    look_backward=1,
    look_forward=0,
    seqlen=512 * 2,
)

text = torch.randn(1, 1024, 512)
img = torch.randn(1, 3, 224, 224)
audio = torch.randn(1, 100)
video = torch.randn(1, 3, 16, 224, 224)
mask = torch.ones(1, 2057).bool()

out = model(text, img, audio, video, mask=mask)
print(out)
