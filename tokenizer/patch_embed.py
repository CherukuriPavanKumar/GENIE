import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Chops each frame into non-overlapping patches, represents each as a
    vector of `embed_dim` numbers. [B,T,C,H,W] -> [B,T,P,E]"""

    def __init__(self, frame_size=64, patch_size=4, in_channels=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x.reshape(B, T, x.shape[1], x.shape[2])
        return x