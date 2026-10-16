"""Environment variables: single source of truth (torch-free).

Every `os.environ` read in the runtime goes through this module, so a
variable rename (or a new one) touches exactly one file. Test doubles
and scripts use the `*_VAR` name constants, never literals.
"""
import os
import sys

assert "torch" not in sys.modules, "torch must never be imported here"

#: Model directory override (checked for the version's acoustic file).
MODELS_DIR_VAR = "SILERO_ONNX_MODELS"
#: Tokenizer cache root override (default `~/.cache/silero_onnx`).
CACHE_VAR = "SILERO_ONNX_CACHE"
#: Disable CLI colors (standard convention; also auto-off on pipes).
NO_COLOR_VAR = "NO_COLOR"


def models_dir():
    """Raw `SILERO_ONNX_MODELS` value, or `""` when unset.

    Explicit user configuration: honored by both the library and the
    CLI. Callers validate the value against what they need (the engine
    checks the version's acoustic file); a wrong path fails loudly
    downstream, never silently falls back.
    """
    return os.environ.get(MODELS_DIR_VAR) or ""


def cache_dir():
    """Raw `SILERO_ONNX_CACHE` override, or `None` (caller applies default)."""
    return os.environ.get(CACHE_VAR) or None


def colors_enabled():
    """ANSI colors only for live terminals that allow them."""
    if os.environ.get(NO_COLOR_VAR):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False
