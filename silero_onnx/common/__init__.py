"""Version-neutral services shared by frontends and strategies."""
from .ssml_lowering import SegmentPlan, SsmlLoweringService, SsmlPlan

__all__ = ["SsmlLoweringService", "SsmlPlan", "SegmentPlan"]
