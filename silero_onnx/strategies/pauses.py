"""Backward-compatible alias for the shared break machinery.

New code imports from ``strategies.common.breaks``.
"""
from .common.breaks import (BreakProcessor, PauseService, apply_fx_pauses,
                            synthesize_with_breaks)

__all__ = ["BreakProcessor", "PauseService", "synthesize_with_breaks",
           "apply_fx_pauses"]
