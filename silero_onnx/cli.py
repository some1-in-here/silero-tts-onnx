"""CLI entry point (thin re-export; implementation lives in engine.py)."""
from .engine import ColorHelpFormatter, examples_text, main

__all__ = ["main", "examples_text", "ColorHelpFormatter"]
