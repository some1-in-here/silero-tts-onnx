"""Strategy contract + common audio tail and request validation.

A concrete strategy owns only the version-specific session sequence and
feed assembly. Language, frontend, sessions and the model-version object
are injected, so a strategy never constructs them itself.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..frontend import normalize
from .common.feeds import FeedBuilder

SAMPLE_RATE = 48000


class AbstractTTSStrategy(ABC):
    """Template for version-specific TTS inference."""

    version_id = ""

    def __init__(self, model_dir, data_dir, sessions, frontend, version=None,
                 language=None):
        self._model_dir = model_dir
        self._data_dir = data_dir
        self._frontend = frontend
        self._sessions = sessions
        if version is None:
            from ..models import ModelRegistry
            version = ModelRegistry.get(self.version_id)
        self._version = version
        self._language = language or getattr(frontend, "language", None)
        self._feeds = FeedBuilder(version)

    @property
    def version(self):
        return self._version

    @property
    def language(self):
        return self._language

    @property
    def speakers(self):
        return sorted(self._frontend.speakers)

    def synthesize(self, text, speaker="xenia", sample_rate=SAMPLE_RATE,
                   ssml=False):
        """Validate the request, then run the version-specific pipeline."""
        if not text or not text.strip():
            raise SystemExit("silero-onnx: error: empty text: nothing to synthesize")
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(
                f"silero-onnx: error: only sample_rate={SAMPLE_RATE} "
                f"is supported (got {sample_rate})")
        if ssml and not self._version.supports_ssml:
            raise ValueError(
                f"SSML is not supported for version "
                f"'{self._version.version_id}'")
        if speaker not in self._frontend.speakers:
            raise ValueError(f"unknown speaker '{speaker}', want one of "
                             f"{sorted(self._frontend.speakers)}")
        return self._synthesize(text, speaker, ssml=ssml)

    @abstractmethod
    def _synthesize(self, text, speaker, ssml=False):
        """Version-specific inference (sessions + feed assembly)."""

    def _normalize(self, text):
        fn = getattr(self._frontend, "normalize", None)
        return fn(text) if callable(fn) else normalize(text)

    def _stress(self, text):
        fn = getattr(self._frontend, "stress", None)
        if callable(fn):
            return fn(text)
        if self._language is not None:
            return self._language.apply_stress(text)
        return text

    def synthesize_to_file(self, text, path, speaker="xenia",
                           sample_rate=SAMPLE_RATE, ssml=False):
        """Like synthesize(), but writes a WAV file; returns its path."""
        audio, _dur = self.synthesize(text, speaker, sample_rate, ssml=ssml)
        import os
        outdir = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(outdir or ".", exist_ok=True)
        except Exception as e:
            raise SystemExit(f"silero-onnx: error: cannot create '{outdir}': {e}")
        try:
            import soundfile as sf
        except ImportError:
            raise SystemExit("silero-onnx: error: soundfile is required "
                             "(pip install soundfile)")
        try:
            sf.write(path, audio, sample_rate)
        except Exception as e:
            raise SystemExit(f"silero-onnx: error: cannot write '{path}': {e}")
        return path

    def _vocode(self, mel):
        """Shared tail: spectral parts -> waveform (v5 path only)."""
        mag, x, y = self._sessions["vocoder"].run(
            None, {"mel": np.asarray(mel, dtype=np.float32)})
        return self._sessions["istft"].run(
            None, {"mag": np.asarray(mag, dtype=np.float32),
                   "x": np.asarray(x, dtype=np.float32),
                   "y": np.asarray(y, dtype=np.float32)})[0]


#: Historical name kept for callers that imported the ABC directly.
TTSStrategy = AbstractTTSStrategy

__all__ = ["AbstractTTSStrategy", "TTSStrategy", "SAMPLE_RATE"]
