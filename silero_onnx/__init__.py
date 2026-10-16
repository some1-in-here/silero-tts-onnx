"""silero_onnx: torch-free Silero TTS v5 RU (text -> ONNX -> WAV).

Public API is intentionally tiny; everything else is internal::

    from silero_onnx import SileroTTS
    tts = SileroTTS(model_dir="path/to/models", version="v5_ru")
    audio = tts.synthesize("Привет, мир!", speaker="xenia")  # float32 mono, 48kHz
"""
from .engine import main
from .strategies import SileroTTS

__all__ = ["SileroTTS", "main"]
