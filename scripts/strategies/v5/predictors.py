"""Clean predictor ports: SeriesPredictor (dur/pitch inner), wrappers, LengthRegulator.

Mirrors jit_forward_model.py TransformerSeriesPredictor (:177-194),
JitDurPredictor (:235-236), JitOldPitchPredictor (:412-413) and
forward_layers.py LengthRegulator (:29-34).
- emotion_embedding is dead weight (absent from all graphs) and omitted.
- coef_speakers voice-mixing branch is ported (standard ops) but unused
  in practice (always None); trace bakes the taken branch.
- LengthRegulator has NO weights; it mutates dur in place exactly like
  the original (index_put_ clamp) — callers rely on post-call values.
"""
from typing import Optional

import torch

from encoder import PositionalEncoding
from fftblock import FFTBlock
from layernorm import ExportLayerNorm


class SeriesPredictor(torch.nn.Module):
    # n_speakers has NO default (unlike dims, it comes from the
    # checkpoint): constructing without it is a TypeError, never a
    # silent 5-voice module. Dimensions keep defaults - they are
    # architecture, verified by shape asserts at transplant.
    def __init__(self, n_speakers, n_symbols=47, d_model=64, n_heads=2,
                 n_layers=4, ffn_dim=64, kernel=9, p_drop=0.1, pe_max=5000):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, d_model)
        self.speaker_embedding = torch.nn.Embedding(n_speakers, d_model)
        self.pos = PositionalEncoding(d_model, pe_max, p_drop)
        self.layers = torch.nn.ModuleList(
            [FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop) for _ in range(n_layers)])
        self.norm = ExportLayerNorm(d_model)
        self.lin = torch.nn.Linear(d_model, 1)

    def _body(self, x, mask):
        # x: (B,T,E) -> transpose -> pos -> transpose -> layers -> norm
        h = self.pos(x.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)

    def forward(self, x: torch.Tensor, speaker_ids: torch.Tensor,
                mask: Optional[torch.Tensor],
                alpha: float,
                coef_speakers: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = self.embedding(x)                      # (B,T,E)
        t = e.size(1)
        if coef_speakers is None:
            spk = self.speaker_embedding(speaker_ids)          # (1,E)
        else:
            w = self.speaker_embedding.weight                  # (S,E)
            cw = coef_speakers.reshape(-1, 1)                  # (S,1)
            mixed = (w * cw)[coef_speakers != 0].sum(dim=0, keepdim=False)
            spk = mixed.unsqueeze(0)                           # (1,E)
        spk = spk.unsqueeze(1).repeat(1, t, 1)                 # (1,T,E)
        h = self._body(e + spk, mask)
        return self.lin(h) / alpha                             # (B,T,1)


class DurPredictor(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.inner = SeriesPredictor(**kw)

    def forward(self, x, speaker_ids, mask, alpha, coef_speakers=None):
        return self.inner(x, speaker_ids, mask, alpha, coef_speakers).squeeze(-1)


class PitchPredictor(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.inner = SeriesPredictor(**kw)

    def forward(self, text, speaker_ids, mask, alpha):
        # JitOldPitchPredictor: pitch_pred(...) then transpose(1,2)
        return self.inner(text, speaker_ids, mask, alpha, None).transpose(1, 2)


class LengthRegulator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        # x: (B,T,E), dur: (B,T) float. Mirrors forward_layers.py exactly,
        # including the in-place clamp of negative durations.
        # Single-utterance form (B=1, the only mode of this pipeline):
        # the general B>1 loop + aten::pad_sequence of the original is
        # omitted (pad_sequence has no ONNX lowering; B>1 never occurs).
        dur[dur < 0] = 0
        # int64 (not the original int32): dynamo/onnxscript lowers the
        # upsampling through Range, and ORT rejects Range with mixed
        # int32/int64 inputs. repeat_interleave accepts int64 identically.
        reps = (dur[0] + 0.5).to(torch.int64)
        r = torch.repeat_interleave(x[0], reps, dim=0)
        return r.reshape(1, -1, x.size(-1))
