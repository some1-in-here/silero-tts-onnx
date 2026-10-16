"""Clean encoder: token embedding + sinusoidal pos-encoding + 4xFFTBlock + norm.

Mirrors ForwardTransformer.forward (fastpitch_layers.py:308-319) and
PositionalEncoding.forward (:55-56):
  x(B,T,E) -> transpose(0,1) -> x + scale*pe[0:T] -> dropout
  -> transpose(0,1) -> 4x FFTBlock(mask, style=None) -> LayerNorm
"""
from typing import Optional

import torch

from fftblock import FFTBlock
from layernorm import ExportLayerNorm


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=128, max_len=5000, p_drop=0.1):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.empty(1))
        self.register_buffer("pe", torch.empty(max_len, 1, d_model))
        self.dropout = torch.nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,E); pe[:T] is (T,1,E), broadcasts over batch
        t = x.size(0)
        return self.dropout(x + self.scale * self.pe[:t])


class Encoder(torch.nn.Module):
    def __init__(self, n_symbols=47, d_model=128, n_heads=2, n_layers=4,
                 ffn_dim=512, kernel=9, p_drop=0.1, pe_max=5000):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, d_model)
        self.pos = PositionalEncoding(d_model, pe_max, p_drop)
        self.layers = torch.nn.ModuleList(
            [FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop) for _ in range(n_layers)])
        self.norm = ExportLayerNorm(d_model)

    def forward(self, ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(ids)
        x = self.pos(x.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
