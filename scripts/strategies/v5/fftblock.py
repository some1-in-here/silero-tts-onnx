"""Clean FFTBlock port (encoder layer, use_linear=False, pitch_pred=False).

Mirrors tacotron2/fastpitch_layers.py FFTBlock.forward exactly
(verified against stored TorchScript graph of encoder/layers/0):
  attn(src,src,src,mask) -> drop -> +src -> norm1 -> transpose(1,2)
  -> conv1(k9,pad4) -> relu -> conv2(k1) -> drop -> +residual
  -> transpose(1,2) -> norm2
style input of the original is dead (never used in body) and omitted.
use_linear/pitch_pred branches are False in all inspected layers;
other variants (if any) will be ported separately when reached.
"""
from typing import Optional

import torch

from slowmha import SlowMHA
from layernorm import ExportLayerNorm


class FFTBlock(torch.nn.Module):
    def __init__(self, d_model=128, n_heads=2, ffn_dim=512, kernel=9, p_drop=0.1):
        super().__init__()
        self.attn = SlowMHA(d_model, n_heads)
        self.drop1 = torch.nn.Dropout(p_drop)
        self.norm1 = ExportLayerNorm(d_model)
        self.conv1 = torch.nn.Conv1d(d_model, ffn_dim, kernel, padding=kernel // 2)
        self.conv2 = torch.nn.Conv1d(ffn_dim, d_model, 1)
        self.drop2 = torch.nn.Dropout(p_drop)
        self.norm2 = ExportLayerNorm(d_model)

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a, _w = self.attn(src, src, src, mask, True, None, True)
        h = self.norm1(src + self.drop1(a))
        t0 = h.transpose(1, 2)
        t1 = self.drop2(self.conv2(torch.relu(self.conv1(t0))))
        t2 = t0 + t1
        return self.norm2(t2.transpose(1, 2))
