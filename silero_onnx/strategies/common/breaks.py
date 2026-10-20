"""Shared SSML <break> machinery (two acoustic passes + fx_pauses).

Moved from ``strategies/pauses.py`` unchanged. ``BreakProcessor`` is
version-independent; the only model-specific parameter is the mel
silence fill dumped from the checkpoint (passed by the strategy).
"""
import numpy as np


class BreakProcessor:
    """Two-pass break forcing and mel pause surgery for one version."""

    def __init__(self, sil_value):
        self.sil_value = float(sil_value)

    def synthesize(self, acoustic, feeds, breaks):
        """Two-pass acoustic run with forced pauses -> (mel, dur)."""
        mel, dur = acoustic.run(None, feeds)
        rates = np.array(feeds["durs_rate"], dtype=np.float32)
        for pos, frames in breaks:
            base = float(dur[0, pos])
            rates[0, pos] = (base / frames) if (frames > 0 and base > 0) else 1.0
        feeds2 = dict(feeds)
        feeds2["durs_rate"] = rates
        mel, dur = acoustic.run(None, feeds2)
        return self.apply(mel, dur, breaks), dur

    def apply(self, mel, dur, breaks):
        """Numpy port of upstream fx_pauses (post-decoder mel surgery)."""
        mel = np.array(mel, dtype=np.float32, copy=True)
        row = np.asarray(dur[0], dtype=np.float64)
        for pos, _frames in breaks:
            start = int(row[:pos].sum())
            sdur = int(row[pos])
            if sdur > 10:
                mel[:, :, start + 10:start + sdur] = self.sil_value
            else:
                mel[:, :, start:start + sdur] = self.sil_value
        return mel


class PauseService(BreakProcessor):
    """Strategy-facing pause service.

    ``BreakProcessor`` remains the low-level two-pass implementation;
    strategies depend on this named service so pause handling has one
    explicit seam in the version-independent layer.
    """


def synthesize_with_breaks(acoustic, feeds, breaks, sil_value):
    """Compatibility wrapper (call sites from the pre-refactor code)."""
    return BreakProcessor(sil_value).synthesize(acoustic, feeds, breaks)


def apply_fx_pauses(mel, dur, breaks, sil_value):
    """Compatibility wrapper for the numpy fx_pauses port."""
    return BreakProcessor(sil_value).apply(mel, dur, breaks)


__all__ = ["BreakProcessor", "PauseService", "synthesize_with_breaks",
           "apply_fx_pauses"]
