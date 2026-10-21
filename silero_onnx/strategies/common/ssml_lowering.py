"""Backward-compatible SSML lowering import path.

The implementation lives in :mod:`silero_onnx.common.ssml_lowering`.
Runtime strategies import the canonical module directly; this shim keeps
old external imports working and contains no implementation of its own.
"""
from ...common.ssml_lowering import (SegmentPlan, SsmlLoweringService,
                                     SsmlPlan)

__all__ = ["SsmlLoweringService", "SsmlPlan", "SegmentPlan"]
