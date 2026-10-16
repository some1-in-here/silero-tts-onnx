"""v5_ru strategy: the current pipeline, moved here unchanged.

Plain text (1 acoustic pass), SSML prosody (1 pass), SSML breaks
(two-pass duration forcing + numpy fx_pauses port). See engine.py
history for the original implementation; behavior is pinned by the
full test suite running with version="v5_ru".
"""
import os

import numpy as np

from ..base import SAMPLE_RATE, TTSStrategy


class Strategy_V5(TTSStrategy):
    """Single-pass (+SSML two-pass) inference on the v5_ru model set."""

    def __init__(self, model_dir, data_dir, sessions, frontend):
        super().__init__(model_dir, data_dir, sessions, frontend)
        try:
            import json
            with open(os.path.join(data_dir, "pause.json"),
                      encoding="utf-8") as f:
                self._sil_value = float(json.load(f)["sil_value"])
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(
                f"silero-onnx: error: cannot load pause data from "
                f"'{data_dir}': {e}")

    @property
    def speakers(self):
        """Available voice names (e.g. aidar baya kseniya eugene xenia)."""
        return sorted(self._frontend.speakers)

    def synthesize(self, text, speaker="xenia", sample_rate=SAMPLE_RATE,
                   ssml=False):
        """Russian text -> (mono float32 audio, durations array).

        ssml=True parses <speak> XML (break/prosody/p/s tags, see
        silero_onnx/ssml.py). Prosody needs one acoustic pass; <break>
        pauses need a second pass with forced durations (the graph has no
        symb_durs input) — see _synthesize_breaks.
        """
        if not text or not text.strip():
            raise SystemExit("silero-onnx: error: empty text: nothing to synthesize")
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(
                f"silero-onnx: error: only sample_rate={SAMPLE_RATE} is supported "
                f"(got {sample_rate})")
        acoustic = self._sessions["acoustic"]
        if ssml:
            feeds, breaks = self._frontend.prepare_ssml(text, speaker)
        else:
            feeds = self._frontend.prepare(text, speaker)
            breaks = []
        if breaks:
            mel, dur = self._synthesize_breaks(acoustic, feeds, breaks,
                                               self._sil_value)
        else:
            mel, dur = acoustic.run(None, feeds)
        audio = self._vocode(mel)
        return np.ascontiguousarray(audio.squeeze()), dur

    @staticmethod
    def _synthesize_breaks(acoustic, feeds, breaks, sil_value):
        """Two-pass pause forcing + mel silencing (no graph changes needed).

        Pass 1 runs with the SSML rates to get base durations; pass 2
        re-runs with rates rewritten so round(dur/rate) == requested
        frames at each break token (mirrors the upstream in-graph
        dur[sid] = sdur override, which also lands pre-len_reg); then
        _apply_fx_pauses flattens the pause regions in mel (mirrors the
        upstream post-decoder fx_pauses).
        Known edge (documented, not silent): a base duration of 0
        cannot be lifted by rate scaling (0/x == 0) — that pause then
        undershoots. Boundary letters virtually never predict 0.
        """
        mel, dur = acoustic.run(None, feeds)
        rates = np.array(feeds["durs_rate"], dtype=np.float32)
        for pos, frames in breaks:
            base = float(dur[0, pos])
            rates[0, pos] = (base / frames) if (frames > 0 and base > 0) else 1.0
        feeds2 = dict(feeds)
        feeds2["durs_rate"] = rates
        mel, dur = acoustic.run(None, feeds2)
        mel = Strategy_V5._apply_fx_pauses(mel, dur, breaks, sil_value)
        return mel, dur

    @staticmethod
    def _apply_fx_pauses(mel, dur, breaks, sil_value):
        """Numpy port of the upstream post-decoder fx_pauses mel surgery.

        For each (pos, frames): the sum of final durations before pos
        locates the pause in mel frames; the region is flattened to
        sil_value (upstream keeps the first 10 frames for pauses over
        10 frames). Runs on the (1,192,F) mel between the acoustic and
        vocoder sessions.
        Documented deviation: breaks on comma positions additionally
        re-predict durations upstream (batch-2 support path with
        pause-swapped tokens); here they take the same override+silence
        as other positions. Measured parity impact is small (see the
        SSML parity probe); exact support-path port would need a graph
        re-export for batch-2 duration prediction.
        """
        mel = np.array(mel, dtype=np.float32, copy=True)
        row = np.asarray(dur[0], dtype=np.float64)
        for pos, sdur in breaks:
            start = int(row[:pos].sum())
            if sdur > 10:
                mel[:, :, start + 10:start + sdur] = sil_value
            else:
                mel[:, :, start:start + sdur] = sil_value
        return mel
