"""Clean v5_6 vocoder port (JitV4 48kHz path only).

Mirrors jit_vocoder.py JitV4/JitV4Backbone/JitConvNeXtBlock + heads.py
ISTFTHead for sample_rate=48000, coef=0.0, inference=True:
  h = backbone(mel); (mag, phase) = chunk(head.out(h)); audio =
  RealISTFT(clamp(exp(mag), max 100), cos(phase), sin(phase)).
The PQMF resamplers (24k/8k paths) are NOT ported: sr bakes to 48000,
exactly the engine operating point (the standalone session keeps the
same contract as the fused main tail).
Complex tensors never appear: spec splits into real/imag parts feeding
the proven float64-core RealISTFT (same n_fft/hop/win/padding as v5).
"""
import torch

from istft_model import RealISTFT
from layernorm import ExportLayerNorm


class ConvNeXtBlock56(torch.nn.Module):
    def __init__(self, dim=512, inter=1536, kernel=7, pad=3):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(dim, dim, kernel, padding=pad,
                                      groups=dim)
        self.norm = ExportLayerNorm(dim)
        self.pwconv1 = torch.nn.Linear(dim, inter)
        self.act = torch.nn.GELU(approximate="none")
        self.pwconv2 = torch.nn.Linear(inter, dim)
        self.gamma = torch.nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x).transpose(1, 2)
        h = self.norm(h)
        h = self.pwconv2(self.act(self.pwconv1(h)))
        return x + (h * self.gamma).transpose(1, 2)


class VocoderBackbone56(torch.nn.Module):
    def __init__(self, mel_dim=192, dim=512, n_blocks=8):
        super().__init__()
        self.embed = torch.nn.Conv1d(mel_dim, dim, 7, padding=3)
        self.norm = ExportLayerNorm(dim)
        self.blocks = torch.nn.ModuleList(
            [ConvNeXtBlock56(dim) for _ in range(n_blocks)])
        self.final_norm = ExportLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).transpose(1, 2)
        h = self.norm(h).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.final_norm(h.transpose(1, 2))


class Vocoder56(torch.nn.Module):
    def __init__(self, dim=512, n_bins=1201):
        super().__init__()
        self.backbone = VocoderBackbone56()
        self.out = torch.nn.Linear(dim, 2 * n_bins)
        self.istft = RealISTFT()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.backbone(mel)
        mag, phase = torch.chunk(self.out(h).transpose(1, 2), 2, dim=1)
        mag = torch.clamp(torch.exp(mag), max=100.0)
        audio = self.istft(mag, torch.cos(phase), torch.sin(phase))
        return audio
