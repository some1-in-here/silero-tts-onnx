"""Version-aware acoustic feed assembly + durs_rate validation."""
import numpy as np


class FeedBuilder:
    """Builds ONNX feed dicts from a version's declared input names.

    The builder is the only place that knows which optional inputs a
    graph expects. Missing required inputs fail loudly; optional stress
    inputs become zeros when the version does not support stress.
    """

    def __init__(self, version):
        self._version = version
        self._inputs = tuple(version.acoustic_inputs())

    @property
    def version_id(self):
        return self._version.version_id

    @staticmethod
    def validate_durs_rate(durs_rate):
        """Every rate coefficient must be finite and strictly positive."""
        arr = np.asarray(durs_rate)
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError("durs_rate must be finite")
        if arr.size and not np.all(arr > 0):
            raise ValueError("durs_rate must be > 0")
        return durs_rate

    def acoustic_feeds(self, *, sequence, speaker_ids, durs_rate=None,
                       pitch_coefs=None, gt_durs=None, gt_pitch=None,
                       type_ids=None, focus_mask=None):
        """Assemble a feed dict for the version's acoustic session."""
        n = int(np.asarray(sequence).shape[1])
        if durs_rate is None:
            durs_rate = np.ones((1, n), dtype=np.float32)
        self.validate_durs_rate(durs_rate)
        if "pitch_coefs" in self._inputs and pitch_coefs is None:
            pitch_coefs = np.ones((1, n), dtype=np.float32)
        if "focus_mask" in self._inputs:
            if focus_mask is None or not self._version.supports_stress:
                focus_mask = np.zeros((1, n), dtype=np.int64)
        values = {
            "sequence": sequence,
            "speaker_ids": speaker_ids,
            "durs_rate": durs_rate,
            "pitch_coefs": pitch_coefs,
            "gt_durs": gt_durs,
            "gt_pitch": gt_pitch,
            "type_ids": type_ids,
            "focus_mask": focus_mask,
        }
        missing = [name for name in self._inputs if values.get(name) is None]
        if missing:
            raise ValueError(
                f"version '{self.version_id}' requires acoustic inputs "
                f"{missing}; the strategy did not supply them")
        return {name: values[name] for name in self._inputs}


def zero_focus_mask(n):
    """A (1, n) zero focus mask for versions without stress support."""
    return np.zeros((1, int(n)), dtype=np.int64)


__all__ = ["FeedBuilder", "zero_focus_mask"]
