"""Clean HourGlass decoder port (+ helpers).

Mirrors tacotron2/fastpitch_layers.py HourGlassTransformer,
AvgPoolDownsample, LinearUpsample, pad_to_multiple exactly:
  x(B,T,E) -> transpose -> pos -> transpose -> pre(mask,None)
  -> pad_to_multiple(T -> ceil(T/3)*3) -> residual=clone
  -> avgpool-down (transpose-conv-transpose, k=3,s=3)   [masks: None path]
  -> shorten0(down) [output DISCARDED, quirk of original lines 59-60]
  -> shorten1(down) -> upsample proj(128->384) + view(B,-1,128)
  -> +residual -> [:,:L] -> post(mask=None) -> norm
Masks: decoder is always called with (None, None); tensor-mask branches
are ported for structure but bake to the None path at trace.
"""
from typing import Optional

import torch

from encoder import PositionalEncoding
from fftblock import FFTBlock
from layernorm import ExportLayerNorm


def pad_to_multiple(t: torch.Tensor, multiple: int, value: float = 0.0) -> torch.Tensor:
    # Branch-free variant of ts fastpitch_layers.py pad_to_multiple: padding
    # with remainder 0 is a no-op, so always padding is exactly equivalent
    # and keeps dynamo export dynamic for every T (a Python `if` on shapes
    # would bake one side at trace/export time).
    shape = t.shape
    seq = shape[-1] if len(shape) == 2 else shape[-2]
    rem = (multiple - seq % multiple) % multiple
    if len(shape) == 2:
        return torch.nn.functional.pad(t, [0, rem], value=value)
    return torch.nn.functional.pad(t, [0, 0, 0, rem], value=value)


class AvgPoolDownsample(torch.nn.Module):
    def __init__(self, kernel=3, stride=3):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool1d(kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            return self.avg_pool(x)
        h = x.transpose(1, 2)
        h = self.avg_pool(h)
        return h.transpose(1, 2)


class LinearUpsample(torch.nn.Module):
    def __init__(self, d_model=128, factor=3):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model * factor)
        self.dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return h.view(h.size(0), -1, self.dim)


class HourGlassDecoder(torch.nn.Module):
    def __init__(self, d_model=128, n_heads=2, ffn_dim=512, kernel=9,
                 p_drop=0.1, pe_max=5000, shorten=3):
        super().__init__()
        self.pos = PositionalEncoding(d_model, pe_max, p_drop)
        self.pre = FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop)
        self.shorten_factor = shorten
        self.downsample = AvgPoolDownsample(shorten, shorten)
        self.short0 = FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop)
        self.short1 = FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop)
        self.upsample = LinearUpsample(d_model, shorten)
        self.post = FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop)
        self.norm = ExportLayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        L = x.size(1)
        h = self.pos(x.transpose(0, 1)).transpose(0, 1)
        h = self.pre(h, mask)
        h = pad_to_multiple(h, self.shorten_factor, 0.0)
        res = h.clone()
        d = self.downsample(h)
        if mask is None:
            dmask = None
        else:
            mf = mask.to(torch.float32)
            mf = pad_to_multiple(mf, self.shorten_factor, 0.0)
            dmask = (self.downsample(mf) > 0)
        _dead = self.short0(d, dmask)  # output discarded, as in original
        s = self.short1(d, dmask)
        u = self.upsample(s)
        h = (u + res)[:, :L]
        h = self.post(h, mask)
        return self.norm(h)
