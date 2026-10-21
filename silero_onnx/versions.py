"""Backward-compatible version view (thin wrapper over models/).

New code should import from ``silero_onnx.models``. This module keeps
the historical ``MODEL_SETS`` / ``VERSIONS`` / ``DEFAULT_VERSION``
surface stable for the CLI, scripts and tests.
"""
from .models import MODEL_SPECS

#: Registered model sets that declare artifacts. Empty placeholder specs
#: stay out of the compatibility/CLI surface until they get a model set.
MODEL_SETS = {vid: dict(spec.model_files)
              for vid, spec in MODEL_SPECS.items() if spec.model_files}

VERSIONS = tuple(MODEL_SETS)

DEFAULT_VERSION = "v5_6_ru"

__all__ = ["MODEL_SETS", "VERSIONS", "DEFAULT_VERSION"]
