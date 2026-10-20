"""v5 build blocks (canonical import point).

``acoustic.py`` holds the clean tacotron port; ``build_acoustic.py``
transplants the upstream checkpoint into it. This thin module mirrors
the ``scripts/strategies/v5_6/build.py`` naming so both version tracks
have the same shape without moving proven code.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.dirname(THIS_DIR)
SCRIPTS_DIR = os.path.dirname(STRATEGIES_DIR)
for _p in (THIS_DIR, STRATEGIES_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acoustic import AcousticClean  # noqa: E402
from build_acoustic import build_acoustic  # noqa: E402

__all__ = ["AcousticClean", "build_acoustic"]
