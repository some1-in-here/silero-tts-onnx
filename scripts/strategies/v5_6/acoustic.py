"""Clean v5_6 acoustic port (text -> mel, dur).

Tacotron_V56 mirrors JitMultiForward.forward (energy path omitted — the
main call hardcodes energy_hat=None, so it is dead):
  e = emb(ids); h = transformer(e, mask) + spk; h += pitch_proj(pitch)
  (pitch_emb_type="default" bakes the add, as in v5);
  expanded = len_reg(h, dur); mel = lin(decoder(expanded)).transpose.
Reuses Encoder / HourGlassDecoder / LengthRegulator / ExportLayerNorm.

Acoustic_V56 is the exportable half of the fused torch main (split at the
mel boundary, which is a real intermediate of the scripted graph). It
takes the raw predictor pitch plus per-token pitch_coefs and applies the
checkpoint's update_pitch_coef before Tacotron, so <prosody pitch> has a
real channel in the always-gt runtime. Outputs (mel, dur_hat7). The
vocoder tail ships as its own session.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DIR = os.path.join(os.path.dirname(THIS_DIR), "v5")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (THIS_DIR, V5_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from decoder import HourGlassDecoder  # noqa: E402
from encoder import Encoder  # noqa: E402
from predictors import LengthRegulator  # noqa: E402


# Shared duration post-processing. Kept as module functions (not static
# methods on Main_V56) so Acoustic_V56 can use them without importing the
# fused main and creating a cycle.
def _cap_sos(dur: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    # if dur[0,0] > cap: dur[0,0] = cap  <=>  minimum (exact)
    dur = dur.clone()
    dur[0][0] = torch.minimum(dur[0][0], cap.to(dur.dtype))
    return dur


def _fx_durs(dur: torch.Tensor, cap: torch.Tensor,
             emph: torch.Tensor) -> torch.Tensor:
    # Branch-free fx_durs (ts jit_model :288-319): cap_sos, then
    # emph-gated clamp-up of positions [1:n] (n = min(5, T-1)) to 5,
    # [-1] capped at 7, [-2] forced to 13, [-3] capped at 13.
    dur = _cap_sos(dur, cap)
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


class Tacotron_V56(torch.nn.Module):
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


class Acoustic_V56(torch.nn.Module):
    """Text -> (mel, dur): branch-free control flow + Tacotron_V56."""

    def __init__(self, tacotron: Tacotron_V56, mean_std):
        super().__init__()
        if mean_std is None:
            raise ValueError(
                "Acoustic_V56: mean_std must come from the checkpoint "
                "(no default: it shifts pitch prosody)")
        table = torch.tensor(list(mean_std), dtype=torch.float32)
        if table.numel() != tacotron.spk_emb.num_embeddings:
            raise ValueError(
                "Acoustic_V56: mean_std has %d speakers, Tacotron has %d"
                % (table.numel(), tacotron.spk_emb.num_embeddings))
        self.tacotron = tacotron
        self.register_buffer("mean_std_table", table)

    def _update_pitch_coef(self, norm_pitch, pitch_coef, speaker_ids):
        """Exact update_pitch_coef from the scripted v5_6 main.

        ``gt_pitch`` bypasses this step inside the scripted forward, but
        the runtime always feeds gt, so the acoustic port applies it here
        to keep <prosody pitch> functional. The formula is the upstream
        one: zero tiny predicted values, scale, add the speaker-dependent
        mean/std shift, and keep exact zeros silent.
        """
        norm_pitch = norm_pitch.clone()
        norm_pitch[torch.abs(norm_pitch) < 0.001] = 0.0
        out = norm_pitch * pitch_coef
        idx = torch.where(pitch_coef[0] == 0)[0]
        pitch_coef = pitch_coef.clone()
        pitch_coef[:, idx] = 1.0
        shift = (pitch_coef - 1.) * self.mean_std_table[
            speaker_ids.reshape(-1)[0]]
        shift[norm_pitch[0] == 0] = 0.0
        return out + shift

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                durs_rate: torch.Tensor, gt_durs: torch.Tensor,
                gt_pitch: torch.Tensor, pitch_coefs: torch.Tensor,
                type_ids: torch.Tensor, focus_mask: torch.Tensor) -> tuple:
        del focus_mask  # feeds only the standalone predictors, never here
        mask = torch.transpose((torch.transpose(sequence, 0, 1) == 0), 0, 1)
        emph = type_ids[0][0] >= 6
        cap = torch.where(emph, torch.tensor(20), torch.tensor(5))
        dur = gt_durs.clone()
        dur[dur < 0] = 0
        dur = torch.round(dur)
        dur = _cap_sos(dur, cap)
        dur = torch.round(dur / durs_rate)
        dur = _fx_durs(dur, cap, emph)
        pitch = self._update_pitch_coef(
            gt_pitch, pitch_coefs[0].unsqueeze(0), speaker_ids)
        mel = self.tacotron(sequence, speaker_ids, mask, dur, pitch)
        return mel, dur


__all__ = ["Acoustic_V56", "Tacotron_V56"]
