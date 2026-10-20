"""Clean v5_6 predictor ports (TransformerSeriesPredictor + wrappers).

Mirrors jit_forward_model.py TransformerSeriesPredictor (:99-184),
JitDurPredictor (:77-98, squeeze(-1)) and JitPitchPredictorIC (:185-202,
transpose(1,2)):
  e = emb(x) + spk; e += type_emb(type_ids) if use_type;
  e += focus_emb(focus_mask) if use_focus;
  (emphasis input is never fed by the package — the module is ported
  and transplanted, but no emphasis input exists; use_emphasis bakes
  False, exactly the runtime operating point);
  transformer(mask) -> norm -> lin -> (/1.0 baked, exact noop).

Reuses the proven v5 blocks (FFTBlock/use_linear=False/pitch_pred=False,
PositionalEncoding, ExportLayerNorm, SlowMHA inside FFTBlock): measured
layer flags and dims are identical to v5 (conv (64,64,9), attn (192,64)
2 heads, pe (5000,1,64)), only the type/focus adds are new.
"""
import sys
import os

import torch

# Path bootstrap: scripts/ (_common) + scripts/strategies/v5/
# (shared v5 blocks: encoder/decoder/fftblock/layernorm/
# predictors/istft_model) + this dir (v5_6 ports).
_HERE56 = os.path.dirname(os.path.abspath(__file__))
_STRAT56 = os.path.dirname(_HERE56)
for _p56 in (os.path.dirname(_STRAT56), os.path.join(_STRAT56, "v5"), _HERE56):
    if _p56 not in sys.path:
        sys.path.insert(0, _p56)
del _HERE56, _STRAT56, _p56

from encoder import PositionalEncoding
from fftblock import FFTBlock
from layernorm import ExportLayerNorm


class SeriesPredictor_V56(torch.nn.Module):
    # n_speakers has NO default (unlike dims, it comes from the
    # checkpoint): constructing without it is a TypeError, never a
    # silent 5-voice module.
    def __init__(self, n_speakers, n_symbols=47, n_types=12, n_focus=4,
                 n_emphasis=2, d_model=64, n_heads=2, n_layers=4, ffn_dim=64,
                 kernel=9, p_drop=0.1, pe_max=5000):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, d_model)
        self.speaker_embedding = torch.nn.Embedding(n_speakers, d_model)
        self.type_embedding = torch.nn.Embedding(n_types, d_model)
        self.focus_embedding = torch.nn.Embedding(n_focus, d_model)
        self.emphasis_embedding = torch.nn.Embedding(n_emphasis, d_model)
        self.pos = PositionalEncoding(d_model, pe_max, p_drop)
        self.layers = torch.nn.ModuleList(
            [FFTBlock(d_model, n_heads, ffn_dim, kernel, p_drop)
             for _ in range(n_layers)])
        self.norm = ExportLayerNorm(d_model)
        self.lin = torch.nn.Linear(d_model, 1)
        # Baked from the live module flags at transplant (python bools, so
        # dynamo keeps them static — same pattern as pitch_emb_type in v5).
        self.use_type = True
        self.use_focus = True
        self.use_emphasis = False

    def forward(self, x: torch.Tensor, speaker_ids: torch.Tensor,
                mask: torch.Tensor, type_ids: torch.Tensor,
                focus_mask: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)
        t = e.size(1)
        spk = self.speaker_embedding(speaker_ids)
        h = e + spk.unsqueeze(1).repeat(1, t, 1)
        if self.use_type:
            h = h + self.type_embedding(type_ids)
        if self.use_focus:
            h = h + self.focus_embedding(focus_mask)
        # use_emphasis is always False at runtime (package never feeds
        # emphasis); kept as a static branch for structure parity.
        h = self.pos(h.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            h = layer(h, mask)
        return self.lin(self.norm(h))


class DurPredictor_V56(torch.nn.Module):
    def __init__(self, n_types, n_speakers):
        super().__init__()
        self.inner = SeriesPredictor_V56(n_types=n_types,
                                          n_speakers=n_speakers)

    def forward(self, text, speaker_ids, mask, type_ids, focus_mask):
        return self.inner(text, speaker_ids, mask, type_ids,
                          focus_mask).squeeze(-1)


class PitchPredictor_V56(torch.nn.Module):
    def __init__(self, n_types, n_speakers):
        super().__init__()
        self.inner = SeriesPredictor_V56(n_types=n_types,
                                          n_speakers=n_speakers)

    def forward(self, text, speaker_ids, mask, type_ids, focus_mask):
        return self.inner(text, speaker_ids, mask, type_ids,
                          focus_mask).transpose(1, 2)
