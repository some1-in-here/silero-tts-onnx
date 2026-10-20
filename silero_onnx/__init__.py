"""silero_onnx: torch-free Silero TTS (text -> ONNX -> WAV).

Public API is intentionally small::

    from silero_onnx import SileroTTS
    tts = SileroTTS(model_dir="path/to/models", version="v5_ru")
    audio, _dur = tts.synthesize("Привет, мир!", speaker="xenia")
"""
from .engine import EngineFactory, SileroTTS, main

__all__ = ["SileroTTS", "EngineFactory", "main"]
