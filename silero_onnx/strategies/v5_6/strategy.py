"""v5_6 strategy: emphasis-aware text -> ONNX(x4) -> audio.

Pipeline (the documented engine operating point):
  markers (**/*/intensity) -> clean text -> normalize -> stress ->
  focus re-marking -> token ids (+sos/eos) -> type_ids/focus_mask ->
  dur/pitch predictor sessions -> gt_durs=exp(pred)-1, gt_pitch raw ->
  acoustic session (mel, dur) -> vocoder session -> audio.

Text algorithms live in silero_onnx/emphasis.py (exact torch port,
parity-probed); this file is orchestration + session I/O only.
Deviations from upstream (all loud/documented):
- full-sequence predictor gt (no windowed refinement): multi-sentence
  focus/emphasis prosody differs from torch apply_tts there (measured
  in export e2e-info); single-span emphasis matches near-exactly.
- SSML is rejected (v5_6 has no pitch_coefs input and no break
  protocol yet): plain text with markers only.
- focus words that cannot be re-matched after normalization
  (e.g. digits expanded to several words) are silently ignored,
  exactly like upstream (its matcher has no assert either).
"""
import sys

import numpy as np

assert "torch" not in sys.modules, "torch must never be imported here"

from collections import namedtuple

from . import emphasis as E
from ...frontend import normalize
from ..base import SAMPLE_RATE, TTSStrategy

V56Feeds = namedtuple("V56Feeds", ("sequence", "speaker_ids", "type_ids",
                                  "base_ids", "focus_mask", "durs_rate"))


class Strategy_V5_6(TTSStrategy):
    """Torch-free Russian TTS (v5_6 voices, 48 kHz, CPU)."""

    version = "v5_6_ru"

    def __init__(self, model_dir, data_dir, sessions, frontend):
        super().__init__(model_dir, data_dir, sessions, frontend)
        # Speaker-table validation is structural, not nominal: ids must
        # be dense 0..N-1 with unique non-empty names (no hardcoded
        # voice list anywhere — a new upstream voice flows through with
        # zero code changes once models + speakers.json are refreshed).
        # Unknown names still fail per call (ValueError, see synthesize).
        ids = sorted(frontend.speakers.values())
        if not ids or ids != list(range(len(ids))):
            raise SystemExit(
                f"silero-onnx: error: corrupt speaker table for v5_6_ru: "
                f"ids must be dense 0..N-1, got {ids}")
        if any(not n for n in frontend.speakers):
            raise SystemExit(
                "silero-onnx: error: corrupt speaker table for v5_6_ru: "
                "empty voice name")
        table = frontend.symbol_to_id
        self._id_to_symbol = [s for s, _ in sorted(table.items(),
                                                  key=lambda kv: kv[1])]
        if len(self._id_to_symbol) != max(table.values()) + 1:
            raise SystemExit(
                "silero-onnx: error: corrupt symbol table "
                "(ids are not dense)")

    @property
    def speakers(self):
        return sorted(self._frontend.speakers)

    def synthesize(self, text, speaker="xenia",
                   sample_rate=SAMPLE_RATE, ssml=False):
        """Russian text (+ **/*/intensity) -> (mono float32 audio, dur)."""
        if not text or not text.strip():
            raise SystemExit(
                "silero-onnx: error: empty text: nothing to synthesize")
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(
                f"silero-onnx: error: only sample_rate={SAMPLE_RATE} "
                f"is supported (got {sample_rate})")
        if ssml:
            raise SystemExit(
                "silero-onnx: error: SSML is not supported for v5_6_ru "
                "(use plain text with **emphasis**, *focus* and "
                "<intensity=N> markers)")
        if speaker not in self._frontend.speakers:
            raise ValueError(f"unknown speaker '{speaker}', want one of "
                             f"{sorted(self._frontend.speakers)}")
        f = self._prepare(text, speaker)
        seq, spk, ti, base, fm = (f.sequence, f.speaker_ids, f.type_ids,
                                 f.base_ids, f.focus_mask)
        dur_raw = self._sessions["dur_predictor"].run(None, {
            "text": seq, "speaker_ids": spk, "mask": (seq == 0),
            "type_ids": ti, "focus_mask": fm})[0]
        pit = self._sessions["pitch_predictor"].run(None, {
            "text": seq, "speaker_ids": spk, "mask": (seq == 0),
            "type_ids": base, "focus_mask": fm})[0]
        gt_d = (np.exp(dur_raw).astype(np.float32) - 1.0).astype(np.float32)
        mel, dur = self._sessions["acoustic"].run(None, {
            "sequence": seq, "speaker_ids": spk, "durs_rate": f.durs_rate,
            "gt_durs": gt_d.astype(np.float32),
            "gt_pitch": pit.astype(np.float32),
            "type_ids": ti, "focus_mask": fm})
        audio = self._sessions["vocoder"].run(
            None, {"mel": mel.astype(np.float32)})[0]
        return np.ascontiguousarray(audio.squeeze()), dur

    def _prepare(self, text, speaker):
        """Text -> V56Feeds namedtuple (numpy in / numpy out, no sessions).

        Named fields (not positional) so callers cannot swap focus_mask
        with base_ids (an ORT Gather OOB is what a swap costs).
        """
        marked, intensity = E.extract_intensity(text)
        marked = E.strip_markup_tags(marked)
        marked, emph_idx, _has_emphasis = E.split_emphasis(marked)
        has_focus = "*" in marked
        clean = marked.replace("*", "")
        if not clean.strip():
            raise SystemExit(
                "silero-onnx: error: empty text: nothing to synthesize")
        norm_marked = normalize(marked)
        ids, stressed = self._frontend.text_to_ids(clean)
        seq = np.array([ids], dtype=np.int64)
        n = seq.shape[1]
        spk = np.array([self._frontend.speakers[speaker]], dtype=np.int64)
        table = self._frontend.symbol_to_id
        if has_focus:
            accented = E.apply_focus_marks(stressed, norm_marked)
            _c, fm = E.parse_focus_from_text(accented, n, table,
                                             intensity=intensity)
        else:
            fm = np.zeros((1, n), dtype=np.int64)
        spans, sents = E.token_sent_spans(ids, self._id_to_symbol)
        types = E.classify_text(clean).split("|")
        if len(types) != len(spans):
            types = [E.classify_sentence(s.replace("+", ""))
                     for s in sents]
        es = E.emph_spans_from_focus(spans, fm[0].tolist(), emph_idx)
        ti = E.build_type_ids_tokens(ids, types, es, self._id_to_symbol)
        base = (ti % len(E.TYPE2ID)).astype(np.int64)
        rate = np.ones((1, n), dtype=np.float32)
        return V56Feeds(sequence=seq, speaker_ids=spk, type_ids=ti,
                        base_ids=base, focus_mask=fm, durs_rate=rate)
