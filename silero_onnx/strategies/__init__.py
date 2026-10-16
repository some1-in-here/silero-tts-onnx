"""Strategy registry, factory, and the public SileroTTS facade.

Version selection lives in exactly one place: create_strategy().
Unknown versions fail with the supported list; missing assets fail
with the expected filenames (no silent fallbacks anywhere).
"""
import os

from .. import env
from ..versions import DEFAULT_VERSION, MODEL_SETS
from .base import TTSStrategy
from .v5.strategy import Strategy_V5
from .v5_6.strategy import Strategy_V5_6

__all__ = ["TTSStrategy", "Strategy_V5", "Strategy_V5_6",
           "create_strategy", "SileroTTS"]

_STRATEGIES = {
    "v5_ru": Strategy_V5,
    "v5_6_ru": Strategy_V5_6,
}


def _open_session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR only: hide benign optimizer chatter
    try:
        return ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        raise SystemExit(f"silero-onnx: error: cannot load {path}: {e}")


def create_strategy(version=DEFAULT_VERSION, model_dir=None, data_dir=None):
    """Build the inference strategy for a Silero model version."""
    if version not in MODEL_SETS or version not in _STRATEGIES:
        raise ValueError(f"unknown version {version!r}, want one of "
                         f"{sorted(_STRATEGIES)}")
    from ..engine import _dev_data_dir, _env_models_dir
    from ..frontend import TorchFreeFrontend
    files = MODEL_SETS[version]
    if model_dir is None:
        # Library rule: no <repo>/models/onnx autodiscovery here (only the CLI
        # autoresolves the local build output). Either the env var or an
        # explicit path is required.
        model_dir = _env_models_dir(version)
    if model_dir is None:
        raise SystemExit(
            "silero-onnx: error: no model_dir given: pass "
            "model_dir='path/to/models' explicitly (or set "
            f"{env.MODELS_DIR_VAR}). The library never auto-discovers "
            f"<repo>/models/onnx/ - only the silero-onnx CLI does that "
            f"(version '{version}' needs: {', '.join(files.values())})")
    if data_dir is None:
        data_dir = _dev_data_dir()
    if not os.path.isdir(model_dir):
        raise SystemExit(
            "silero-onnx: error: no ONNX models found for version "
            f"'{version}' (expected {', '.join(files.values())} in "
            "model_dir). Build them with python scripts/export_all.py "
            "(v5_ru) or python scripts/strategies/v5_6/export.py (v5_6_ru) "
            "(see scripts/README.md), pass model_dir=... explicitly, or "
            f"set {env.MODELS_DIR_VAR}. ({env.MODELS_DIR_VAR}=%r, "
            "model_dir=%r)" % (env.models_dir(),
                               model_dir))
    try:
        frontend = TorchFreeFrontend(models_dir=model_dir, data_dir=data_dir)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(
            f"silero-onnx: error: cannot load frontend data from "
            f"'{data_dir}': {e}")
    sessions = {}
    for key, fname in files.items():
        base = os.path.join(model_dir, fname)
        if not os.path.isfile(base):
            raise SystemExit(
                f"silero-onnx: error: missing model file {base} "
                f"(version '{version}' needs: {', '.join(files.values())})")
        sessions[key] = _open_session(base)
    return _STRATEGIES[version](model_dir, data_dir, sessions, frontend)


class SileroTTS:
    """Version-selected facade (default v5_6_ru; see versions.py).

    v5_6 assets are built by scripts/strategies/v5_6/export.py (see
    scripts/strategies/v5_6/README.md); without them, pass version="v5_ru"
    explicitly (or --model-version v5_ru on the CLI).

    Library rule: pass model_dir=... explicitly (or set
    SILERO_ONNX_MODELS) — <repo>/models/onnx/ autodiscovery lives in the CLI
    only, never here.
    """

    def __init__(self, model_dir=None, data_dir=None,
                 version=DEFAULT_VERSION):
        self._impl = create_strategy(version, model_dir, data_dir)
        self._version = version

    @property
    def speakers(self):
        return self._impl.speakers

    def synthesize(self, *args, **kwargs):
        return self._impl.synthesize(*args, **kwargs)

    def synthesize_to_file(self, *args, **kwargs):
        return self._impl.synthesize_to_file(*args, **kwargs)
