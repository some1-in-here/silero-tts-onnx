"""Shared strategy contract + common audio tail.

A strategy owns version-specific inference behind one interface; the
tail (vocoder + ISTFT + WAV writing) is identical for every version
and lives here exactly once.
"""
from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 48000


class TTSStrategy(ABC):
    """Abstract TTS inference strategy (see strategies/v5.py, v56.py)."""

    def __init__(self, model_dir, data_dir, sessions, frontend):
        self._model_dir = model_dir
        self._frontend = frontend
        self._sessions = sessions

    @property
    @abstractmethod
    def speakers(self):
        """Available voice names."""

    @abstractmethod
    def synthesize(self, text, speaker="xenia", sample_rate=SAMPLE_RATE,
                   ssml=False):
        """Text -> (mono float32 audio, durations array)."""

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
        """Shared tail: spectral parts -> waveform (numpy in / out)."""
        mag, x, y = self._sessions["vocoder"].run(
            None, {"mel": np.asarray(mel, dtype=np.float32)})
        return self._sessions["istft"].run(
            None, {"mag": np.asarray(mag, dtype=np.float32),
                   "x": np.asarray(x, dtype=np.float32),
                   "y": np.asarray(y, dtype=np.float32)})[0]
