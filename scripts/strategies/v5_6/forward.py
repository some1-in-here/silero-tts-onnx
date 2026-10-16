"""Clean v5_6 tacotron + fused main ports.

Tacotron56 mirrors JitMultiForward.forward (energy path omitted — the
main call hardcodes energy_hat=None, so it is dead):
  e = emb(ids); h = transformer(e, mask) + spk; h += pitch_proj(pitch)
  (pitch_emb_type="default" bakes the add, as in v5);
  expanded = len_reg(h, dur); mel = lin(decoder(expanded)).transpose.
Reuses Encoder / HourGlassDecoder / LengthRegulator / ExportLayerNorm.

Main56 mirrors MultiTTSModel.forward for the engine operating point
(ONLY: symb_durs=None, gt_durs+gt_pitch always fed, sr=48000,
device=cpu, accent_word=-1, question=False, emphasis=None):
  mask=(seq==0); emph=(type_ids[0,0]>=6) [tensor bool];
  dur = round(clamp0(gt)); cap_sos; dur = round(dur/rate);
  fx_durs branch-free; mel = tacotron(...); audio = vocoder(mel).
All value-dependent branches are rewritten with torch.where/minimum
(the v5 _clamp_dur_tail discipline): `if bool(tensor)` would bake one
side at export and silently diverge on other texts.
Deviations (documented, numerically exact):
- gt inputs are CLONED first: the scripted main mutates caller gt in
  place (index_put_); the port leaves caller tensors untouched.
- The punkt-zero loop is provably dead (loop bound min(...,len-1 slice
  of dim-0-size-1 cumsum...) is 0 for every input) and omitted, as in v5.
- B=1 single-utterance by construction ([0] indexing throughout, as v5).
"""
import torch

from decoder import HourGlassDecoder
from encoder import Encoder
from predictors import LengthRegulator


class Tacotron56(torch.nn.Module):
    def __init__(self, n_speakers):
        super().__init__()
        self.encoder = Encoder()
        self.spk_emb = torch.nn.Embedding(n_speakers, 128)
        self.pitch_proj = torch.nn.Conv1d(1, 128, 3, padding=1)
        self.len_reg = LengthRegulator()
        self.decoder = HourGlassDecoder()
        self.lin = torch.nn.Linear(128, 192)
        self.pitch_emb_type = "default"
        self.pitch_strength = 1.0

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                mask: torch.Tensor, dur_hat: torch.Tensor,
                pitch_hat: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(sequence, mask)
        spk = self.spk_emb(speaker_ids).unsqueeze(1).repeat(1, enc.size(1), 1)
        h = enc + spk
        pp = self.pitch_proj(pitch_hat).transpose(1, 2)
        if self.pitch_emb_type == "default":
            h = h + pp * self.pitch_strength
        expanded = self.len_reg(h, dur_hat)
        expanded = expanded.reshape(1, -1, 128)
        dec = self.decoder(expanded, None)
        return self.lin(dec).transpose(1, 2)


class Acoustic56(torch.nn.Module):
    """Text -> (mel, dur): branch-free control flow + Tacotron56.

    This is the exportable half of the fused torch main (split at the
    mel boundary, which is a real intermediate in the scripted graph):
    same inputs minus pitch_coefs (ignored under gt, see Main56 note),
    outputs (mel, dur_hat7). The vocoder tail ships as its own session.
    """

    def __init__(self, tacotron: Tacotron56):
        super().__init__()
        self.tacotron = tacotron

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                durs_rate: torch.Tensor, gt_durs: torch.Tensor,
                gt_pitch: torch.Tensor, type_ids: torch.Tensor,
                focus_mask: torch.Tensor) -> tuple:
        del focus_mask  # feeds only the standalone predictors, never here
        mask = torch.transpose((torch.transpose(sequence, 0, 1) == 0), 0, 1)
        emph = type_ids[0][0] >= 6
        cap = torch.where(emph, torch.tensor(20), torch.tensor(5))
        dur = gt_durs.clone()
        dur[dur < 0] = 0
        dur = torch.round(dur)
        dur = Main56._cap_sos(dur, cap)
        dur = torch.round(dur / durs_rate)
        dur = Main56._fx_durs(dur, cap, emph)
        mel = self.tacotron(sequence, speaker_ids, mask, dur, gt_pitch)
        return mel, dur


class Main56(torch.nn.Module):
    def __init__(self, acoustic: Acoustic56, vocoder: torch.nn.Module):
        super().__init__()
        self.acoustic = acoustic
        self.vocoder = vocoder

    @staticmethod
    def _cap_sos(dur: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        # if dur[0,0] > cap: dur[0,0] = cap  <=>  minimum (exact)
        dur = dur.clone()
        dur[0][0] = torch.minimum(dur[0][0], cap.to(dur.dtype))
        return dur

    @staticmethod
    def _fx_durs(dur: torch.Tensor, cap: torch.Tensor,
                 emph: torch.Tensor) -> torch.Tensor:
        # Branch-free fx_durs (ts jit_model :288-319): cap_sos, then
        # emph-gated clamp-up of positions [1:n] (n = min(5, T-1)) to 5,
        # [-1] capped at 7, [-2] forced to 13, [-3] capped at 13.
        dur = Main56._cap_sos(dur, cap)
        t = dur.size(1)
        pos = torch.arange(t, device=dur.device)
        m = emph & (pos >= 1) & (pos < 5) & (pos < t - 1)
        dur = dur.clone()
        dur[0] = torch.where(m, torch.maximum(dur[0], torch.tensor(
            5., dtype=dur.dtype, device=dur.device)), dur[0])
        dur[0][-1] = torch.minimum(dur[0][-1], torch.tensor(
            7., dtype=dur.dtype, device=dur.device))
        dur[0][-2] = torch.tensor(13., dtype=dur.dtype, device=dur.device)
        dur[0][-3] = torch.minimum(dur[0][-3], torch.tensor(
            13., dtype=dur.dtype, device=dur.device))
        return dur

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                durs_rate: torch.Tensor, gt_durs: torch.Tensor,
                gt_pitch: torch.Tensor, type_ids: torch.Tensor,
                focus_mask: torch.Tensor) -> tuple:
        # Always-gt operating point (mirrors the scripted graph with gt
        # present): pitch_coefs have NO input — the scripted graph ignores
        # them when gt_pitch is fed (gt bypasses update_pitch_coef), and
        # durs_rate still applies. focus_mask feeds only the standalone
        # predictor sessions, never the main. Torch-parity wrapper only
        # (the exportable units are Acoustic56 + Vocoder56 separately:
        # fold-based ISTFT does not dynamo-export inside the fused graph).
        mel, dur = self.acoustic(sequence, speaker_ids, durs_rate, gt_durs,
                                 gt_pitch, type_ids, focus_mask)
        audio = self.vocoder(mel)
        return audio, dur
