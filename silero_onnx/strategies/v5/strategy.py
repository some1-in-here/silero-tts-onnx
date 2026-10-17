"""v5_ru strategy: the current pipeline, moved here unchanged.

Plain text (1 acoustic pass), SSML prosody (1 pass), SSML breaks
(two-pass duration forcing + numpy fx_pauses port, now shared with the
v5_6 track in strategies/pauses.py). Behavior is pinned by the full
test suite running with version="v5_ru".
"""
import os

import numpy as np

from ..base import SAMPLE_RATE, TTSStrategy
from ..pauses import synthesize_with_breaks


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
        symb_durs input) — see strategies/pauses.py.
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
            mel, dur = synthesize_with_breaks(acoustic, feeds, breaks,
                                              self._sil_value)
        else:
            mel, dur = acoustic.run(None, feeds)
        audio = self._vocode(mel)
        return np.ascontiguousarray(audio.squeeze()), dur

