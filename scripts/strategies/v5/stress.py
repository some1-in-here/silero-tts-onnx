"""v5 stress export entry point (canonical module name).

``export_stress.py`` owns the TorchScript homograph-solver export;
this module keeps the version-track structure consistent.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.dirname(THIS_DIR)
SCRIPTS_DIR = os.path.dirname(STRATEGIES_DIR)
for _p in (THIS_DIR, STRATEGIES_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_stress import main as export_stress  # noqa: E402

__all__ = ["export_stress"]
