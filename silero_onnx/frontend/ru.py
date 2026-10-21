"""Russian text frontend (``TextFrontend_Ru``), language injected via DI."""
import numpy as np

from ..common.ssml_lowering import SsmlLoweringService
from ..languages.base import add_sos_eos, tokenize
from ..languages.ru import Language_Ru
from ..languages.ru.symbols import DATA_DIR
from .base import AbstractTextFrontend
from .pipeline import TextPipeline


class TextFrontend(AbstractTextFrontend):
    """Generic frontend: works with any ``AbstractLanguage``.

    ``supports_pitch`` is a capability supplied by the caller (normally
    ``EngineFactory`` from the selected model version). The frontend never
    reads a ``ModelSpec`` itself; it only decides whether to emit the
    ``pitch_coefs`` input the version expects.
    """

    def __init__(self, language, data_dir=None, supports_pitch=True):
        super().__init__(language)
        self._data_dir = data_dir
        self._supports_pitch = bool(supports_pitch)
        self._pipeline = TextPipeline(language)
        _symbols, self._symbol_to_id = language.get_symbol_table()
        self._speakers = language.get_speakers()

    @property
    def speakers(self):
        return dict(self._speakers)

    @property
    def symbol_to_id(self):
        return self._symbol_to_id

    def normalize(self, text):
        return self._pipeline.normalize(text)

    def stress(self, text):
        return self._pipeline.stress(text)

    def text_to_ids(self, text):
        """Normalize + stress + tokenize -> (ids with sos/eos, stressed)."""
        return self._pipeline.text_to_ids(text, self._symbol_to_id)

    def stress_and_tokenize(self, clean_text):
        """Already-normalized text -> (ids with sos/eos, stressed text)."""
        stressed = self.stress(clean_text)
        ids = add_sos_eos(
            tokenize(stressed, self._symbol_to_id, errors="filter"),
            self._symbol_to_id)
        return ids, stressed

    def _check_speaker(self, speaker):
        if speaker not in self._speakers:
            raise ValueError(f"unknown speaker '{speaker}', want one of "
                             f"{sorted(self._speakers)}")

    def prepare(self, text, speaker="xenia"):
        """Text -> dict of numpy arrays matching acoustic ONNX inputs."""
        self._check_speaker(speaker)
        ids, _stressed = self.text_to_ids(text)
        seq = np.array([ids], dtype=np.int64)
        n = seq.shape[1]
        feeds = {"sequence": seq,
                 "speaker_ids": np.array([self._speakers[speaker]],
                                         dtype=np.int64),
                 "durs_rate": np.ones((1, n), dtype=np.float32)}
        if self._supports_pitch:
            feeds["pitch_coefs"] = np.ones((1, n), dtype=np.float32)
        return feeds

    def prepare_ssml(self, ssml_text, speaker="xenia"):
        """SSML -> (feeds with per-token rate/pitch arrays, break plans).

        Parsing/merging/rate attribution is delegated to the shared
        ``SsmlLoweringService``; this method only assembles the v5 feed
        dict; version-specific feed assembly happens outside the frontend.
        """
        self._check_speaker(speaker)
        service = SsmlLoweringService(normalize=self.normalize,
                                      supports_pitch=self._supports_pitch)
        plan = service.plan(ssml_text)
        if not plan.segments:
            raise ValueError("silero-onnx: error: SSML holds no speakable text")
        stressed = self.stress(" ".join(s.text for s in plan.segments))
        ids = add_sos_eos(
            tokenize(stressed, self._symbol_to_id, errors="filter"),
            self._symbol_to_id)
        rates, breaks = service.attribute(
            stressed, ids, self._symbol_to_id, plan.segments,
            plan.leading_break_ms)
        n = len(ids)
        feeds = {"sequence": np.array([ids], dtype=np.int64),
                 "speaker_ids": np.array([self._speakers[speaker]],
                                         dtype=np.int64),
                 "durs_rate": np.array([rates], dtype=np.float32)}
        if self._supports_pitch:
            pitches = service.attribute_pitch(
                stressed, ids, self._symbol_to_id, plan.segments,
                plan.leading_break_ms)
            feeds["pitch_coefs"] = np.array([pitches], dtype=np.float32)
        return feeds, breaks


class TextFrontend_Ru(TextFrontend):
    """Russian frontend built on ``Language_Ru`` (DI-friendly)."""

    frontend_id = "ru"

    def __init__(self, language=None, models_dir=None, data_dir=None,
                 supports_pitch=True):
        if language is None:
            try:
                language = Language_Ru(models_dir=models_dir,
                                       data_dir=data_dir)
            except (OSError, ValueError, KeyError) as e:
                raise SystemExit(
                    f"silero-onnx: error: cannot load frontend data from "
                    f"'{data_dir or DATA_DIR}': {e}")
        super().__init__(language, data_dir=data_dir,
                         supports_pitch=supports_pitch)


class TorchFreeFrontend(TextFrontend_Ru):
    """Backward-compatible name for the original runtime frontend."""

    def __init__(self, models_dir=None, data_dir=DATA_DIR,
                 supports_pitch=True):
        super().__init__(models_dir=models_dir, data_dir=data_dir,
                         supports_pitch=supports_pitch)


__all__ = ["TextFrontend", "TextFrontend_Ru", "TorchFreeFrontend"]
