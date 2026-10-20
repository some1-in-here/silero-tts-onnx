"""Backward-compatible version view (thin wrapper over models/).

New code should import from ``silero_onnx.models``. This module keeps
the historical ``MODEL_SETS`` / ``VERSIONS`` / ``DEFAULT_VERSION``
surface stable for the CLI, scripts and tests.
"""
from .models import MODEL_SPECS

MODEL_SETS = {vid: dict(spec.model_files)
              for vid, spec in MODEL_SPECS.items()}

VERSIONS = tuple(MODEL_SETS)

DEFAULT_VERSION = "v5_6_ru"

__all__ = ["MODEL_SETS", "VERSIONS", "DEFAULT_VERSION"]
