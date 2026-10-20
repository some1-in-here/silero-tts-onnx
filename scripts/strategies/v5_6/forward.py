"""Clean v5_6 fused main port; acoustic classes live in acoustic.py.

Main_V56 mirrors MultiTTSModel.forward for the engine operating point
(ONLY: symb_durs=None, gt_durs+gt_pitch always fed, sr=48000,
device=cpu, accent_word=-1, question=False, emphasis=None):
  mask=(seq==0); emph=(type_ids[0,0]>=6) [tensor bool];
  dur = round(clamp0(gt)); cap_sos; dur = round(dur/rate);
  pitch = update_pitch_coef(gt_pitch, pitch_coefs, spk);
  fx_durs branch-free; mel = tacotron(...); audio = vocoder(mel).
All value-dependent branches are rewritten with torch.where/minimum
(the v5 _clamp_dur_tail discipline): `if bool(tensor)` would bake one
side at export and silently diverge on other texts.
Deviations (documented, numerically exact):
- gt inputs are CLONED first: the scripted main mutates caller gt in
  place (index_put_); the port leaves caller tensors untouched.
- The scripted forward bypasses update_pitch_coef when gt_pitch is fed.
  The always-gt runtime would therefore lose <prosody pitch>, so the
  clean port applies the same checkpoint formula to the raw predictor
  pitch (validated by L0/L1 and the SSML e2e cases).
- The punkt-zero loop is provably dead (loop bound min(...,len-1 slice
  of dim-0-size-1 cumsum...) is 0 for every input) and omitted, as in v5.
- B=1 single-utterance by construction ([0] indexing throughout, as v5).

The exportable acoustic half is ``Acoustic_V56`` in ``acoustic.py``; this
module re-exports it and ``Tacotron_V56`` for the existing build/export
imports. The fused main is retained as the torch-parity reference.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Both tracks have an `acoustic` module: force the sibling one to win
# even if v5/ was placed on sys.path first.
if THIS_DIR in sys.path:
    sys.path.remove(THIS_DIR)
sys.path.insert(0, THIS_DIR)

import torch  # noqa: E402

from acoustic import Acoustic_V56, Tacotron_V56, _cap_sos, _fx_durs  # noqa: E402


class Main_V56(torch.nn.Module):
    def __init__(self, acoustic: Acoustic_V56, vocoder: torch.nn.Module):
        super().__init__()
        self.acoustic = acoustic
        self.vocoder = vocoder

    # Kept as static methods for the historical parity harness; the
    # implementations are shared with Acoustic_V56 in acoustic.py.
    @staticmethod
    def _cap_sos(dur: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        return _cap_sos(dur, cap)

    @staticmethod
    def _fx_durs(dur: torch.Tensor, cap: torch.Tensor,
                 emph: torch.Tensor) -> torch.Tensor:
        return _fx_durs(dur, cap, emph)

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                durs_rate: torch.Tensor, gt_durs: torch.Tensor,
                gt_pitch: torch.Tensor, pitch_coefs: torch.Tensor,
                type_ids: torch.Tensor,
                focus_mask: torch.Tensor) -> tuple:
        # Always-gt operating point (mirrors the scripted graph with gt
        # present). The acoustic port applies update_pitch_coef to the raw
        # predictor pitch so <prosody pitch> still has a channel; durs_rate
        # still applies. focus_mask feeds only the standalone predictor
        # sessions, never the main. Torch-parity wrapper only (the
        # exportable units are Acoustic_V56 + Vocoder_V56 separately:
        # fold-based ISTFT does not dynamo-export inside the fused graph).
        mel, dur = self.acoustic(sequence, speaker_ids, durs_rate, gt_durs,
                                 gt_pitch, pitch_coefs, type_ids, focus_mask)
        audio = self.vocoder(mel)
        return audio, dur


__all__ = ["Acoustic_V56", "Main_V56", "Tacotron_V56"]
