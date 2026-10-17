"""Shared test configuration: model-set detection + marker registry.

The suite is split into two halves:

- **model-free tests** (frontend, SSML parser, v5_6 text algorithms,
  factory routing, CLI light commands) — no torch, no models, always
  runnable; this is what the regular CI workflow (.github/workflows/
  ci.yml) exercises on every push and pull request;
- **model tests** (synthesis, stress inference, threads, memory, RTF,
  SSML prosody/breaks) — need a built model directory. They are marked
  `requires_models` (v5_ru set) or `requires_v56` (v5_6_ru set) and are
  SKIPPED when the needed set is absent; the run ends with ONE explicit
  summary line (per-test reasons via -rs), so a fresh checkout (or the
  model-less CI job) yields meaningful green runs instead of misleading
  failures - and the CI log is not a wall of skip blocks.

Model resolution mirrors the runtime: SILERO_ONNX_MODELS when set,
otherwise <repo>/models/onnx. The full pipeline (export both sets +
validate + all model tests) runs in the manually triggered E2E
workflow (.github/workflows/e2e.yml) and on release tags.

Both of those pipelines pass --require-models, which turns the
"needed set absent" condition below into a hard error instead of a
skip: a green E2E run therefore proves every model test actually
executed.
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

#: Filled during collection: [(marker, needle, count)] for absent sets, so
#: pytest_terminal_summary can report them in one line instead of turning
#: the CI log into a wall of SKIPPED blocks.
_MISSING_SETS = []


def _models_dir():
    """Where tests look for models (same resolution as the runtime)."""
    return (os.environ.get(env.MODELS_DIR_VAR)
            or os.path.normpath(os.path.join(ROOT, "models", "onnx")))


def pytest_addoption(parser):
    parser.addoption(
        "--require-models", action="store_true", default=False,
        help="error out instead of skipping when a model set needed by "
             "the collected tests is absent (used by e2e.yml and "
             "export_and_release.yml, where a skip would silently mean "
             "'not actually tested')")


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
    require = config.getoption("--require-models")
    del _MISSING_SETS[:]
    for marker, needle in SET_FILES.items():
        if os.path.isfile(os.path.join(where, needle)):
            continue
        if require:
            raise pytest.UsageError(
                f"--require-models: {needle} not found in {where!r}: this "
                "pipeline exists to run the model tests against freshly "
                "built models, so a missing set is an error, not a skip")
        reason = (
            f"needs {needle}: not found in {where!r} - build it once with "
            "`python scripts/export_all.py` or point "
            f"{env.MODELS_DIR_VAR} at a model directory; the full E2E "
            "workflow runs these tests with freshly exported models")
        skip = pytest.mark.skip(reason=reason)
        marked = 0
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
                marked += 1
        if marked:
            _MISSING_SETS.append((marker, needle, marked))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """One explicit line instead of a wall of SKIPPED blocks.

    ci.yml runs pytest without -rs: per-test skip reasons stay available
    on demand, while the run still ends with the total and the pointer to
    where those tests actually run (the Full E2E workflow).
    """
    if not _MISSING_SETS or config.getoption("--require-models"):
        return
    total = sum(marked for _, _, marked in _MISSING_SETS)
    files = ", ".join(needle for _, needle, _ in _MISSING_SETS)
    terminalreporter.write_line(
        f"NOTE: {total} model-dependent tests skipped: {files} not in "
        f"{_models_dir()!r} - build them with `python scripts/export_all.py` "
        f"(or set {env.MODELS_DIR_VAR}); the Full E2E workflow runs them "
        "against freshly exported models")
