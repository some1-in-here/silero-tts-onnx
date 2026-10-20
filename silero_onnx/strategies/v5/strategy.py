"""v5_ru strategy: frontend -> acoustic -> vocoder -> istft.

The strategy is a thin sequence of ONNX session calls. All text rules
live in the injected frontend/language, all SSML lowering in the neutral
``silero_onnx.common`` layer, and the version capabilities in
``Version_V5Ru``.
"""
import numpy as np

from ...languages.base import add_sos_eos, tokenize
from ..base import AbstractTTSStrategy
from ..common.breaks import PauseService
from ...common.ssml_lowering import SsmlLoweringService


class Strategy_V5(AbstractTTSStrategy):
    """Single-pass (+SSML two-pass) inference on the v5_ru model set."""

    version_id = "v5_ru"

    def __init__(self, model_dir, data_dir, sessions, frontend, version=None,
                 language=None):
        super().__init__(model_dir, data_dir, sessions, frontend,
                         version=version, language=language)
        self._breaks = PauseService(self._version.pause_sil_value)

    def _synthesize(self, text, speaker, ssml=False):
        if ssml:
            feeds, breaks = self._prepare_ssml(text, speaker)
        else:
            feeds = self._prepare(text, speaker)
            breaks = []
        acoustic = self._sessions["acoustic"]
        if breaks:
            mel, dur = self._breaks.synthesize(acoustic, feeds, breaks)
        else:
            mel, dur = acoustic.run(None, feeds)
        audio = self._vocode(mel)
        return np.ascontiguousarray(audio.squeeze()), dur

    def _prepare(self, text, speaker):
        """Plain text -> acoustic feeds (validated by FeedBuilder).

        v5_ru has no sentence-type channel, so this path never runs the
        v5_6 question classifier.
        """
        feeds = self._frontend.prepare(text, speaker)
        return self._feeds.acoustic_feeds(**feeds)

    def _prepare_ssml(self, text, speaker):
        """<speak> XML -> (acoustic feeds with rate/pitch arrays, breaks)."""
        if not self._version.supports_ssml:
            raise ValueError(
                f"SSML is not supported for version "
                f"'{self._version.version_id}'")
        service = SsmlLoweringService(normalize=self._normalize,
                                      supports_pitch=self._version.supports_pitch,
                                      version_id=self._version.version_id)
        plan = service.plan(text)
        if not plan.segments:
            raise ValueError("silero-onnx: error: SSML holds no speakable text")
        stressed = self._stress(" ".join(s.text for s in plan.segments))
        table = self._frontend.symbol_to_id
        ids = add_sos_eos(
            tokenize(stressed, table, errors="filter"), table)
        rates, breaks = service.attribute(
            stressed, ids, table, plan.segments, plan.leading_break_ms)
        pitches = service.attribute_pitch(
            stressed, ids, table, plan.segments, plan.leading_break_ms)
        n = len(ids)
        feeds = self._feeds.acoustic_feeds(
            sequence=np.array([ids], dtype=np.int64),
            speaker_ids=np.array([self._frontend.speakers[speaker]],
                                 dtype=np.int64),
            durs_rate=np.array([rates], dtype=np.float32),
            pitch_coefs=np.array([pitches], dtype=np.float32))
        return feeds, breaks


__all__ = ["Strategy_V5"]
