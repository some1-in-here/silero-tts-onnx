"""Export-safe LayerNorm replacement (build-time only).

WHY THIS EXISTS: torch>=2.1x rewrites aten::layer_norm to
onnx::LayerNormalization during classic torch.onnx.export, dropping the
output rank; the immediately following transpose(1,2) then fails with
"ONNX export of transpose for tensor of unknown rank" (observed on
torch 2.14, acoustic graph). Spelling the same math with primitive ops
(mean/var/normalize, all shape-preserving) exports cleanly on any torch.

Numerics: identical formula to torch.nn.LayerNorm (biased variance,
eps=1e-5 default): y = (x - mean)/sqrt(var + eps) * weight + bias.
The transplant in build_acoustic.py copies .weight/.bias by attribute,
so it works unchanged; the acoustic ORT gate (atol=1e-3) re-verifies.
"""
import torch


class ExportLayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.bias = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias
