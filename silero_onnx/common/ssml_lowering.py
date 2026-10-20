"""Unified SSML lowering used by every model version.

The service owns the version-independent half of SSML handling:
- parse + normalize segments (with an optional marker transform),
- merge empty segments and leading breaks,
- validate rate coefficients,
- attribute per-segment rates/pitches to the joint stressed token
  stream and place break plans on the segment-final token.

This module is intentionally outside ``strategies/`` so the text
frontend can use it without depending on the versioned inference layer.
Version-specific strategies still decide how to build their own feeds
(v5 applies pitch_coefs directly; v5_6 applies it to the predictor pitch
via update_pitch_coef); this service only warns when a version declares
that it has no pitch support.
"""
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

from ..ssml import ms_to_frames, parse_ssml


@dataclass(frozen=True)
class SegmentPlan:
    """One speakable SSML segment after normalization/merge."""

    text: str
    rate: float
    break_ms: Optional[int]
    pitch: float = 1.0


@dataclass(frozen=True)
class SsmlPlan:
    """The output of ``SsmlLoweringService.plan``."""

    segments: Tuple[SegmentPlan, ...]
    leading_break_ms: Optional[int] = None
    requested_pitches: Tuple[float, ...] = ()


class SsmlLoweringService:
    """Translate parsed SSML into segment plans + token attribution."""

    def __init__(self, normalize, supports_pitch=True, version_id=None,
                 segment_transform=None):
        self._normalize = normalize
        self._supports_pitch = supports_pitch
        self._version_id = version_id
        self._segment_transform = segment_transform

    def plan(self, ssml_text) -> SsmlPlan:
        segs = parse_ssml(ssml_text)
        requested = tuple(sorted({s["pitch"] for s in segs
                                  if s["pitch"] != 1.0}))
        if requested and not self._supports_pitch:
            vid = self._version_id or "this model version"
            warnings.warn(
                f"silero-onnx: {vid} ignores <prosody pitch=...> "
                f"({', '.join(str(p) for p in requested)}): its acoustic "
                "model has no pitch_coefs channel; <prosody rate> and "
                "<break> work")
        kept, leading = [], None
        for s in segs:
            text = self._normalize(s["text"])
            if self._segment_transform is not None:
                text = self._segment_transform(text)
            rate = float(s["rate"])
            if not rate > 0:
                raise ValueError(
                    "silero-onnx: error: SSML rate must produce durs_rate > 0 "
                    f"(got {rate})")
            if not text:
                if s["break_ms"] is not None:
                    if kept:
                        prev = kept[-1]
                        brk = (s["break_ms"] if prev.break_ms is None
                               else max(prev.break_ms, s["break_ms"]))
                        kept[-1] = SegmentPlan(prev.text, prev.rate, brk,
                                               prev.pitch)
                    elif leading is None:
                        leading = s["break_ms"]
                continue
            kept.append(SegmentPlan(text, rate, s["break_ms"],
                                    float(s["pitch"])))
        return SsmlPlan(tuple(kept), leading, requested)

    def attribute(self, stressed, ids, symbol_to_id, segments,
                  leading_break_ms=None):
        """Joint stressed text -> (per-token durs_rate, break plans)."""
        rates, breaks = self._attribute(
            stressed, ids, symbol_to_id, segments, leading_break_ms,
            values=[s.rate for s in segments])
        return rates, breaks

    def attribute_pitch(self, stressed, ids, symbol_to_id, segments,
                        leading_break_ms=None):
        """Joint stressed text -> per-token pitch coefficients."""
        pitches, _breaks = self._attribute(
            stressed, ids, symbol_to_id, segments, leading_break_ms,
            values=[s.pitch for s in segments], with_breaks=False)
        return pitches

    def _attribute(self, stressed, ids, symbol_to_id, segments,
                   leading_break_ms, values, with_breaks=True):
        if not segments:
            raise ValueError(
                "silero-onnx: error: SSML holds no speakable text")
        counts = [len(s.text.split(" ")) for s in segments]
        words = stressed.split(" ")
        if sum(counts) != len(words):
            raise ValueError(
                "silero-onnx: error: stress changed the SSML word arity "
                f"({sum(counts)} -> {len(words)}): cannot attribute prosody")
        word_vals = [v for v, c in zip(values, counts) for _ in range(c)]
        last_word, acc = [], 0
        for c in counts:
            acc += c
            last_word.append(acc - 1)
        spans, start = [], 0
        for i, ch in enumerate(stressed):
            if ch == " ":
                spans.append((start, i))
                start = i + 1
        spans.append((start, len(stressed)))
        seg_end_char = [spans[k][1] - 1 for k in last_word]
        out, breaks = [word_vals[0]], []
        if leading_break_ms is not None:
            breaks.append((0, ms_to_frames(leading_break_ms)))
        seg, wi = 0, 0
        last = len(word_vals) - 1
        for pos, ch in enumerate(stressed):
            m = ch.lower().replace("—", "–")
            if m in symbol_to_id:
                out.append(word_vals[min(wi, last)])
            if ch == " ":
                wi += 1
            while seg < len(segments) and pos >= seg_end_char[seg]:
                brk = segments[seg].break_ms
                if with_breaks and brk is not None:
                    breaks.append((len(out) - 1, ms_to_frames(brk)))
                seg += 1
        if seg != len(segments):
            raise ValueError(
                "silero-onnx: error: SSML rate attribution lost a segment "
                f"({seg} of {len(segments)} consumed)")
        out.append(word_vals[-1])
        if len(out) != len(ids):
            raise ValueError(
                "silero-onnx: error: SSML rate attribution diverged from "
                f"the token sequence ({len(out)} values vs {len(ids)} tokens)")
        return out, breaks


__all__ = ["SsmlLoweringService", "SsmlPlan", "SegmentPlan"]
