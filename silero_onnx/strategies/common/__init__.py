"""Version-independent strategy services."""
from .breaks import (BreakProcessor, PauseService, apply_fx_pauses,
                     synthesize_with_breaks)
from .feeds import FeedBuilder, zero_focus_mask
from .ssml_lowering import SegmentPlan, SsmlLoweringService, SsmlPlan

__all__ = ["BreakProcessor", "PauseService", "synthesize_with_breaks",
           "apply_fx_pauses", "FeedBuilder", "zero_focus_mask",
           "SsmlLoweringService", "SsmlPlan", "SegmentPlan"]
