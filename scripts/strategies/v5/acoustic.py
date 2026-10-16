"""Clean full acoustic model (tacotron side, vocoder excluded).

Mirrors MultiTTSModel.forward tail (ts jit_model lines 80-226, symb/gt
paths skipped as in real usage) + JitMultiForward (tacotron lines 98-129):
  mask=(seq==0).T; pred=dur(...); dur=round(clamp(exp(pred)-1));
  tail clamps ([0][0]<=5, [0][-1]<=7, [0][-2]=13, [0][-3]<=13);
  pitch=update_pitch_coef(pitch_pred(...), ones); punkt loop (provably
  0 iterations for B=1, guarded by assert); enc=Encoder(ids);
  h=enc+spk; pitch_proj run but not added (pitch_emb_type="default");
  expanded=len_reg(h, dur); dec(expanded); lin; transpose -> mel(B,192,T).
Conventions matching the original: in-place clamps on fresh tensors,
symb_durs=None, durs_rate/pitch_coefs ones, sr unused here (vocoder arg).
"""
from typing import List, Optional

import torch

from decoder import HourGlassDecoder
from encoder import Encoder
from predictors import DurPredictor, LengthRegulator, PitchPredictor

PUNKT = [3, 5, 6, 7, 8, 9, 10, 44, 45]


class AcousticClean(torch.nn.Module):
    def __init__(self, n_speakers, mean_std=None):
        # Both come from the checkpoint (see build_acoustic); the
        # transplant copy_ calls re-verify every shape, so a wrong N
        # fails loudly instead of degrading audio. mean_std has no safe
        # default (it shifts pitch prosody): None raises instead of
        # silently reusing stale numbers.
        if mean_std is None:
            raise ValueError("AcousticClean: mean_std must be read live "
                             "from the checkpoint (no default: it shifts "
                             "pitch prosody)")
        super().__init__()
        self.encoder = Encoder()
        self.dur = DurPredictor(n_speakers=n_speakers)
        self.pitch = PitchPredictor(n_speakers=n_speakers)
        self.spk_emb = torch.nn.Embedding(n_speakers, 128)
        self.pitch_proj = torch.nn.Conv1d(1, 128, 3, padding=1)
        self.len_reg = LengthRegulator()
        self.decoder = HourGlassDecoder()
        self.lin = torch.nn.Linear(128, 192)
        self.pitch_emb_type = "default"
        self.pitch_strength = 1.0
        self.register_buffer(
            "mean_std_table",
            torch.tensor(list(mean_std), dtype=torch.float32))

    @staticmethod
    def _clamp_dur_tail(dur: torch.Tensor) -> torch.Tensor:
        # Branch-free positional clamps (torch.minimum instead of Python ifs)
        # so dynamo export keeps them dynamic for every input (a traced
        # `if bool(tensor)` would bake the taken branch and go wrong on
        # other texts). Mathematically identical to the original if-assigns.
        dur[0][0] = torch.minimum(dur[0][0], torch.tensor(5.))
        dur[0][-1] = torch.minimum(dur[0][-1], torch.tensor(7.))
        dur[0][-2] = 13
        dur[0][-3] = torch.minimum(dur[0][-3], torch.tensor(13.))
        return dur

    def _update_pitch_coef(self, norm_pitch: torch.Tensor, pitch_coef: torch.Tensor,
                           speaker_ids: torch.Tensor) -> torch.Tensor:
        norm_pitch[torch.abs(norm_pitch) < 0.001] = 0.0
        out = norm_pitch * pitch_coef
        idx = torch.where(pitch_coef[0] == 0)[0]
        pitch_coef[:, idx] = 1.0
        shift = (pitch_coef - 1.) * self.mean_std_table[speaker_ids.reshape(-1)[0]]
        shift[norm_pitch[0] == 0] = 0
        return out + shift

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor,
                durs_rate: torch.Tensor, pitch_coefs: torch.Tensor):
        # make_token_len_mask(transpose(sequence,0,1)) -> (B,T) bool
        # (ts jit_model lines 28-30; token id 0 marks padding)
        mask = torch.transpose((torch.transpose(sequence, 0, 1) == 0), 0, 1)
        pred = self.dur(sequence, speaker_ids, mask, 1.0, None)
        dur = torch.exp(pred) - 1
        dur[dur < 0] = 0
        dur = torch.round(dur)
        dur[0][0] = torch.minimum(dur[0][0], torch.tensor(5.))
        dur = torch.round(dur / durs_rate)
        dur = self._clamp_dur_tail(dur)
        p0 = self.pitch(sequence, speaker_ids, mask, 1.0)    # (B,1,T)
        pc = pitch_coefs[0].unsqueeze(0)
        pitch = self._update_pitch_coef(p0, pc, speaker_ids)
        # punkt loop (ts lines 193-209) is provably dead for B=1: the loop
        # bound min(B, B-1, B-1, B) is 0, so the body (which would need
        # int(vector) semantics) never executes. Nothing to replicate.
        enc = self.encoder(sequence, mask)
        spk = self.spk_emb(speaker_ids).unsqueeze(1).repeat(1, enc.size(1), 1)
        h = enc + spk
        pp = self.pitch_proj(pitch).transpose(1, 2)
        # ts jit_forward_model lines 110-111: add pitch iff type == "default"
        if self.pitch_emb_type == "default":
            h = h + pp * self.pitch_strength
        expanded = self.len_reg(h, dur)
        # Static-rank seam for the ONNX exporter: len_reg's output rank is
        # reconstructed by shape inference as unknown in the full graph
        # (dynamic repeat path); pin it explicitly. No-op numerically
        # (tacotron side is always E=128; asserted in transplant).
        expanded = expanded.reshape(1, -1, 128)
        dec = self.decoder(expanded, None)
        mel = self.lin(dec).transpose(1, 2)
        return mel, dur
