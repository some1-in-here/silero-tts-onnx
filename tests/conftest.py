"""Shared test configuration: model-set detection + marker registry.

The suite is split into two halves:

- **model-free tests** (frontend, SSML parser, v5_6 text algorithms,
  factory routing, CLI light commands) — no torch, no models, always
  runnable; this is what the regular CI workflow (.github/workflows/
  ci.yml) exercises on every push and pull request;
- **model tests** (synthesis, stress inference, threads, memory, RTF,
  SSML prosody/breaks) — need a built model directory. They are marked
  `requires_models` (v5_ru set) or `requires_v56` (v5_6_ru set) and are
  SKIPPED with an explicit reason when the needed set is absent, so a
  fresh checkout (or the model-less CI job) yields meaningful green
  runs instead of misleading failures with misleading tracebacks.

Model resolution mirrors the runtime: SILERO_ONNX_MODELS when set,
otherwise <repo>/models/onnx. The full pipeline (export both sets +
validate + all model tests) runs in the manually triggered E2E
workflow (.github/workflows/e2e.yml) and on release tags.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from silero_onnx import env  # noqa: E402

#: marker -> representative file that proves this model set is usable
SET_FILES = {
    "requires_models": "acoustic_v5_ru.onnx",
    "requires_v56": "acoustic_v5_6_ru.onnx",
}


def _models_dir():
    """Where tests look for models (same resolution as the runtime)."""
    return (os.environ.get(env.MODELS_DIR_VAR)
            or os.path.normpath(os.path.join(ROOT, "models", "onnx")))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_models: needs the built v5_ru ONNX model set "
        "(skipped with a reason when absent; see tests/conftest.py)")
    config.addinivalue_line(
        "markers",
        "requires_v56: needs the built v5_6_ru ONNX model set "
        "(skipped with a reason when absent; see tests/conftest.py)")


def pytest_collection_modifyitems(config, items):
    where = _models_dir()
    for marker, needle in SET_FILES.items():
        if os.path.isfile(os.path.join(where, needle)):
            continue
        reason = (
            f"needs {needle}: not found in {where!r} - build it once with "
            "`python scripts/export_all.py` or point "
            f"{env.MODELS_DIR_VAR} at a model directory; the full E2E "
            "workflow runs these tests with freshly exported models")
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
