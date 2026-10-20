"""Backward-compatible SSML lowering import path.

The implementation lives in :mod:`silero_onnx.common.ssml_lowering` so
the frontend can use it without importing the versioned strategies layer.
This module only re-exports the public names for existing callers.
"""
from ...common.ssml_lowering import (SegmentPlan, SsmlLoweringService,
                                     SsmlPlan)

__all__ = ["SsmlLoweringService", "SsmlPlan", "SegmentPlan"]
