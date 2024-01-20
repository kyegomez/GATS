import torch
from gats_torch.main import GATSBlock

# Create a GATSBlock model with the following configuration:
# - Input dimension: 512
# - Number of attention heads: 8
# - Dimension of each attention head: 64
# - Dropout rate: 0.1
# - Window size: 512
# - Causal attention: True
# - Number of tokens to look backward: 1
# - Number of tokens to look forward: 0
# - Sequence length: 512 * 2
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

# Create input tensors for different modalities
text = torch.randn(1, 1024, 512)  # Text input tensor
img = torch.randn(1, 3, 224, 224)  # Image input tensor
audio = torch.randn(1, 100)  # Audio input tensor
video = torch.randn(1, 3, 16, 224, 224)  # Video input tensor
mask = torch.ones(1, 2057).bool()  # Mask tensor for attention

# Pass the input tensors through the GATSBlock model
out, _, _, _ = model(text, img, audio, video, mask=mask)

# Print the output
print(out)
