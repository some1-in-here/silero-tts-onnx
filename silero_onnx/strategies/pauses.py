"""SSML <break> machinery shared by both strategies (torch-free).

Neither acoustic graph exposes upstream's symb_durs dict as an input, so
a requested pause cannot be injected in-graph; both tracks reproduce it
outside the graph in two acoustic passes:

  pass 1  runs with the SSML rates to get base durations;
  pass 2  rewrites only the break positions' rates so round(base/rate)
          lands exactly on the requested frame count. Upstream's
          in-graph `dur[sid] = sdur` override also lands after the rate
          division (before the tail caps), so pass 2 replays the same
          arithmetic the graphs already apply — caps included (see the
          edge notes below).

Then fx_pauses flattens the pause regions in mel (numpy port of the
upstream post-decoder surgery, between the acoustic and the vocoder):
silence fill and pause length come from the FINAL durations, exactly
like upstream (`sdur = int(dur_hat[sid])`, scripted v5_6 forward); pauses
over 10 frames keep their first 10 frames, also upstream's rule. The
fill is model-specific and dumped from the upstream checkpoints, not
guessed: v5_ru -8.0 (data/pause.json), v5_6_ru -5.0 (the
_PAUSE_SIL_VALUE constant in strategies/v5_6/strategy.py).

Documented edges (loud, never silent):
- a base duration of 0 cannot be lifted by rate scaling (0/x == 0):
  that pause is dropped, not shifted (upstream's direct override would
  still lift it — the one case the two-pass rate trick cannot replay);
  boundary letters virtually never predict 0.
- graph-side caps still apply in pass 2 exactly as they do upstream
  after its override: sos cap (v5_ru <= 5; v5_6_ru <= 5, or <= 20 on
  emphasis utterances), v5_6_ru's clamp-up of positions [1,5) to >= 5
  on emphasis utterances, and the shared tail caps ([-1] <= 7, [-2] =
  13, [-3] <= 13). A break landing on a capped position is clamped
  identically to upstream (whose override sits before the same caps).
- comma breaks use the same override + silence. Upstream re-predicts
  neighbor durations through a batch-2 "support" path only when
  gt_durs is absent (v5_ru track feeds gt: documented deviation, see
  DEVELOPMENT.md); v5_6_ru's always-gt operating point skips that path
  upstream too (measured in the scripted forward), so there is no
  deviation there.
"""
import numpy as np


def synthesize_with_breaks(acoustic, feeds, breaks, sil_value):
    """Two-pass acoustic run with forced pauses -> (mel, dur).

    `feeds` is the pass-1 feed dict (its "durs_rate" carries the SSML
    rates); `breaks` is [(token_pos, frames)] from the SSML frontend.
    The returned mel is the post-fx_pauses array for the vocoder.
    """
    mel, dur = acoustic.run(None, feeds)
    rates = np.array(feeds["durs_rate"], dtype=np.float32)
    for pos, frames in breaks:
        base = float(dur[0, pos])
        rates[0, pos] = (base / frames) if (frames > 0 and base > 0) else 1.0
    feeds2 = dict(feeds)
    feeds2["durs_rate"] = rates
    mel, dur = acoustic.run(None, feeds2)
    return apply_fx_pauses(mel, dur, breaks, sil_value), dur


def apply_fx_pauses(mel, dur, breaks, sil_value):
    """Numpy port of upstream fx_pauses (post-decoder mel surgery).

    For each (pos, frames): the sum of final durations before pos
    locates the pause in mel frames and the final duration at pos is the
    region length (upstream: dur_sum / sdur from dur_hat); the region is
    flattened to sil_value, keeping its first 10 frames when longer than
    10. Runs on the (1,192,F) mel between acoustic and vocoder.
    """
    mel = np.array(mel, dtype=np.float32, copy=True)
    row = np.asarray(dur[0], dtype=np.float64)
    for pos, _frames in breaks:
        start = int(row[:pos].sum())
        sdur = int(row[pos])
        if sdur > 10:
            mel[:, :, start + 10:start + sdur] = sil_value
        else:
            mel[:, :, start:start + sdur] = sil_value
    return mel
