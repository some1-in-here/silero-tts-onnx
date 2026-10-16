"""Real-valued ISTFT port (no complex tensors anywhere).

Mirrors the upstream v5 ISTFT (n_fft=2400, hop=600,
win=2400, padding="same") exactly, with irfft replaced by an explicit
real IDFT via precomputed cos/sin bases:
  irfft(spec, n) == real part of IDFT of the Hermitian-mirrored spectrum.
Inputs are the split spectral parts (mag, x, y) with spec = mag*(x + iy),
so the ONNX boundary never sees ComplexFloat (cf. obstacle C).
"""
import math

import torch


class RealISTFT(torch.nn.Module):
    def __init__(self, n_fft=2400, hop_length=600, win_length=2400):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        n = n_fft
        k = torch.arange(n).double()
        # IDFT basis: x[n] = (1/N) sum_k X[k] e^{+i2πkn/N}
        ang = 2 * math.pi * torch.outer(k, k) / n
        self.register_buffer("cos_b", torch.cos(ang))
        self.register_buffer("sin_b", torch.sin(ang))

    def _irfft(self, xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        # xr, xi: (B, N, T) with N = n_fft//2 + 1. Mirror to full spectrum,
        # apply real IDFT, keep real part. Matches torch.fft.irfft(norm="backward").
        # The core runs in float64: 2400-term float32 accumulation would add
        # ~1e-4 error (measured); double keeps it at ~1e-7.
        n = self.n_fft
        xr_full = torch.cat([xr, xr[:, 1:-1, :].flip(1)], dim=1)   # (B,n,T)
        xi_full = torch.cat([xi, -xi[:, 1:-1, :].flip(1)], dim=1)  # conj mirror
        a = xr_full.transpose(1, 2).double()  # (B,T,n)
        b = xi_full.transpose(1, 2).double()
        out_r = (a @ self.cos_b - b @ self.sin_b) / n
        return out_r.transpose(1, 2).float()  # (B,n,T)

    def forward(self, mag: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # mag, x, y: (B,N,T); spec = mag * (x + iy). 48kHz path (no PQMF).
        spec_r = mag * x
        spec_i = mag * y
        ifft = self._irfft(spec_r, spec_i)          # (B, win, T)
        ifft = ifft * self.window[None, :, None]
        b, _, t = ifft.shape
        output_size = (t - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length))[:, 0, 0, :]
        pad = (self.win_length - self.hop_length) // 2
        y = y[:, pad:-pad]
        window_sq = self.window.square().expand(1, t, -1).transpose(1, 2)
        envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length)).squeeze()[pad:-pad]
        return y / envelope
