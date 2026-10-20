"""v5_6 duration predictor (re-export of the clean series port)."""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DIR = os.path.join(os.path.dirname(THIS_DIR), "v5")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (THIS_DIR, V5_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from series import DurPredictor_V56  # noqa: E402

__all__ = ["DurPredictor_V56"]
