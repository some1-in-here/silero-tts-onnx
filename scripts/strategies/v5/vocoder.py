"""v5 vocoder export entry point (canonical module name).

The model is built inline by ``export_vocoder.py`` because it is a
frozen TorchScript submodule wrapper; this module keeps the version
track structure consistent (``vocoder.py`` exists for every version).
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.dirname(THIS_DIR)
SCRIPTS_DIR = os.path.dirname(STRATEGIES_DIR)
for _p in (THIS_DIR, STRATEGIES_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_vocoder import main as export_vocoder  # noqa: E402

__all__ = ["export_vocoder"]
