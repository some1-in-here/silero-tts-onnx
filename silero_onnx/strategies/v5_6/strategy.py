"""v5_6 strategy: emphasis-aware text -> ONNX(x4) -> audio.

Pipeline (the documented engine operating point):
  markers (**/*/intensity) -> clean text -> normalize -> stress ->
  focus re-marking -> token ids (+sos/eos) -> type_ids/focus_mask ->
  dur/pitch predictor sessions -> gt_durs=exp(pred)-1, gt_pitch raw ->
  acoustic session (mel, dur) -> vocoder session -> audio.

SSML (--ssml / ssml=True) lowers onto the same pipeline:
- <prosody rate> fills the durs_rate input exactly like upstream's
  rate tensor (the acoustic graph divides by it after the gt clamp —
  measured at export);
- <break>/<p>/<s> replay upstream's in-graph dur[sid] = sdur override
  through the shared two-pass path (strategies/pauses.py) with the
  v5_6_ru silence fill (_PAUSE_SIL_VALUE below: -5.0, dumped from the
  upstream checkpoint — v5_ru ships its own -8.0 in data/pause.json);
- <prosody pitch> has no channel in this operating point (the acoustic
  model takes gt_pitch from its predictor; there is no pitch_coefs
  input under gt) and warns explicitly — never silently;
- <intensity=N> (upstream's pre-XML pseudo-tag) is stripped before
  parsing exactly like upstream's extract_intensity_from_ssml and
  scales the *focus* mask; the first tag wins for the whole utterance,
  differing values warn (upstream drops them silently); without
  * / ** marks it is inert (upstream audio is identical) and warns —
  never silently.

Text algorithms live in silero_onnx/emphasis.py (exact torch port,
parity-probed); this file is orchestration + session I/O only.
Deviations from upstream (all loud/documented):
- full-sequence predictor gt (no windowed refinement): multi-sentence
  focus/emphasis prosody differs from torch apply_tts there (measured
  in export e2e-info); single-span emphasis matches near-exactly.
- SSML focus re-marking (restore_focus_marks, windowed refinement) is
  NOT ported (see emphasis.py); **/* markers inside SSML segments do
  work through the plain-text pipeline, and <intensity=N> is read
  before XML parsing exactly like upstream - its only upstream home;
  plain text accepts the tag too (superset).
- focus words that cannot be re-matched after normalization
  (e.g. digits expanded to several words) are silently ignored,
  exactly like upstream (its matcher has no assert either).
"""
import sys
import warnings

import numpy as np

assert "torch" not in sys.modules, "torch must never be imported here"

from collections import namedtuple

from . import emphasis as E
from ...frontend import normalize
from ...ssml import ms_to_frames, parse_ssml
from ..base import SAMPLE_RATE, TTSStrategy
from ..pauses import synthesize_with_breaks

V56Feeds = namedtuple("V56Feeds", ("sequence", "speaker_ids", "type_ids",
                                   "base_ids", "focus_mask", "durs_rate"))

# fx_pauses mel fill for SSML pauses (see strategies/pauses.py): a
# version-locked implementation detail of the upstream checkpoint, not
# a user-tunable knob, so it is a constant here — not a data file.
# Dumped from the v5_6_ru TorchScript archive, never guessed; v5_ru
# has its own -8.0 dump in data/pause.json.
_PAUSE_SIL_VALUE = -5.0


def _merge_segments(segments):
    """parse_ssml output -> ([(text, rate, break_ms)], leading_break_ms).

    Mirrors frontend.prepare_ssml's segment merge (v5 track): a segment
    holding no speakable text cannot own prosody — its break folds
    backward into the previous segment's (max of the two), or becomes
    the leading pause when no text precedes it. Text is normalized and
    markup-stripped here so word counts and the merged marked text
    agree by construction.
    """
    kept, leading = [], None
    for s in segments:
        text = E.strip_markup_tags(normalize(s["text"]))
        if not text:
            if s["break_ms"] is not None:
                if kept:
                    pt, pr, pb = kept[-1]
                    kept[-1] = (pt, pr, s["break_ms"] if pb is None
                                else max(pb, s["break_ms"]))
                elif leading is None:
                    leading = s["break_ms"]
            continue
        kept.append((text, s["rate"], s["break_ms"]))
    return kept, leading


def _rate_plan(stressed, n_tokens, table, kept, counts, leading_break):
    """Joint stress output -> (per-token durs_rate, [(token_pos, frames)]).

    Words of `stressed` map to segments by `counts` (per-segment word
    counts of the normalized texts, mirroring prepare_ssml): every
    emitted token takes its word's rate; word separators take the rate
    of the word on their left (the v5 join-space convention). A break
    lands on the last emitted id of its segment — exactly where
    prepare_ssml puts it — so the pause region starts at the segment's
    final phone. Loud guards below, never a silent misalignment.
    """
    word_rates = [r for (_t, r, _b), c in zip(kept, counts)
                  for _ in range(c)]
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
    rates = [word_rates[0]]   # sos: first segment's rate (v5 convention)
    breaks = []
    if leading_break is not None:
        breaks.append((0, ms_to_frames(leading_break)))
    seg, wi = 0, 0
    last = len(word_rates) - 1
    for pos, ch in enumerate(stressed):
        # same filter as frontend.tokenize(errors="filter")
        m = ch.lower().replace("—", "–")
        if m in table:
            rates.append(word_rates[min(wi, last)])
        if ch == " ":
            wi += 1
        while seg < len(kept) and pos >= seg_end_char[seg]:
            brk = kept[seg][2]
            if brk is not None:
                breaks.append((len(rates) - 1, ms_to_frames(brk)))
            seg += 1
    if seg != len(kept):
        raise ValueError(
            "silero-onnx: error: SSML rate attribution lost a segment "
            f"({seg} of {len(kept)} consumed)")
    rates.append(word_rates[-1])   # eos: last segment's rate
    if len(rates) != n_tokens:
        raise ValueError(
            "silero-onnx: error: SSML rate attribution diverged from the "
            f"token sequence ({len(rates)} rates vs {n_tokens} tokens)")
    return rates, breaks


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
        """Russian text (+ **/*/intensity) -> (mono float32 audio, dur).

        ssml=True parses <speak> XML: <prosody rate> and <break>/<p>/<s>
        work (rate via durs_rate, breaks via the two-pass pause path);
        <prosody pitch> warns and is ignored — no pitch channel here.
        """
        if not text or not text.strip():
            raise SystemExit(
                "silero-onnx: error: empty text: nothing to synthesize")
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(
                f"silero-onnx: error: only sample_rate={SAMPLE_RATE} "
                f"is supported (got {sample_rate})")
        if speaker not in self._frontend.speakers:
            raise ValueError(f"unknown speaker '{speaker}', want one of "
                             f"{sorted(self._frontend.speakers)}")
        if ssml:
            f, breaks = self._prepare_ssml(text, speaker)
        else:
            f, breaks = self._prepare(text, speaker), []
        seq, spk, ti, base, fm = (f.sequence, f.speaker_ids, f.type_ids,
                                  f.base_ids, f.focus_mask)
        dur_raw = self._sessions["dur_predictor"].run(None, {
            "text": seq, "speaker_ids": spk, "mask": (seq == 0),
            "type_ids": ti, "focus_mask": fm})[0]
        pit = self._sessions["pitch_predictor"].run(None, {
            "text": seq, "speaker_ids": spk, "mask": (seq == 0),
            "type_ids": base, "focus_mask": fm})[0]
        gt_d = (np.exp(dur_raw).astype(np.float32) - 1.0).astype(np.float32)
        feeds = {"sequence": seq, "speaker_ids": spk,
                 "durs_rate": f.durs_rate, "gt_durs": gt_d,
                 "gt_pitch": pit.astype(np.float32),
                 "type_ids": ti, "focus_mask": fm}
        acoustic = self._sessions["acoustic"]
        if breaks:
            mel, dur = synthesize_with_breaks(acoustic, feeds, breaks,
                                              _PAUSE_SIL_VALUE)
        else:
            mel, dur = acoustic.run(None, feeds)
        audio = self._sessions["vocoder"].run(
            None, {"mel": mel.astype(np.float32)})[0]
        return np.ascontiguousarray(audio.squeeze()), dur

    def _prepare(self, text, speaker):
        """Text -> V56Feeds namedtuple (numpy in / numpy out, no sessions).

        Named fields (not positional) so callers cannot swap focus_mask
        with base_ids (an ORT Gather OOB is what a swap costs).
        """
        return self._build(text, speaker)[0]

    def _prepare_ssml(self, text, speaker):
        """<speak> XML -> (V56Feeds, breaks); rate + pause plans.

        Segments merge into ONE marked text (so **/* keep working) and
        run through the same pipeline as plain text; per-segment
        <prosody rate> values are then re-attributed word by word over
        the joint stress output. Each <break>/<p>/<s> becomes a
        (token_pos, frames) plan for the two-pass pause path; <prosody
        pitch> has no channel in this operating point and warns.
        <intensity=N> is stripped before XML parsing (upstream's
        extract_intensity_from_ssml order) and scales *focus* marks;
        without focus marks it is inert and warns.
        """
        intensity_tag = '<intensity' in text
        if intensity_tag:
            text, intensity = E.extract_intensity(text)
        else:
            intensity = None
        segs = parse_ssml(text)
        requested = sorted({s["pitch"] for s in segs if s["pitch"] != 1.0})
        if requested:
            warnings.warn(
                "silero-onnx: v5_6_ru ignores <prosody pitch=...> "
                f"({', '.join(str(p) for p in requested)}): its acoustic "
                "model takes pitch from the predictor and has no "
                "pitch_coefs input; <prosody rate> and <break> work")
        kept, leading_break = _merge_segments(segs)
        if not kept:
            raise ValueError(
                "silero-onnx: error: SSML holds no speakable text")
        counts = [len(t.split(" ")) for t, _r, _b in kept]
        feeds, stressed = self._build(
            " ".join(t for t, _r, _b in kept), speaker,
            intensity=intensity, intensity_tag=intensity_tag)
        words = stressed.split(" ")
        if sum(counts) != len(words):
            raise ValueError(
                "silero-onnx: error: stress changed the SSML word arity "
                f"({sum(counts)} -> {len(words)}): cannot attribute "
                "prosody")
        rates, breaks = _rate_plan(
            stressed, len(feeds.sequence[0]),
            self._frontend.symbol_to_id, kept, counts, leading_break)
        feeds = feeds._replace(
            durs_rate=np.array([rates], dtype=np.float32))
        return feeds, breaks

    def _build(self, text, speaker, intensity=None, intensity_tag=None):
        """Marked text -> (V56Feeds, stressed); shared plain/SSML path.

        `stressed` is the exact string that produced the token ids (the
        SSML path re-attributes per-segment rates over it, word by
        word). `intensity`/`intensity_tag`: the SSML path strips the
        <intensity=N> pseudo-tag before XML parsing (upstream order) and
        passes both in; plain text leaves them None and the tag is read
        here. The level scales only * / ** marks - without them it is
        inert (upstream audio is identical) and warns, never silent.
        """
        if intensity_tag is None:
            intensity_tag = '<intensity' in text
        marked, extracted = E.extract_intensity(text)
        if intensity is None:
            intensity = extracted
        marked = E.strip_markup_tags(marked)
        marked, emph_idx, _has_emphasis = E.split_emphasis(marked)
        has_focus = "*" in marked
        clean = marked.replace("*", "")
        if not clean.strip():
            raise SystemExit(
                "silero-onnx: error: empty text: nothing to synthesize")
        if intensity_tag and not has_focus:
            warnings.warn(
                f"silero-onnx: <intensity={intensity}> scales *focused* "
                "words and has no effect without *word* / **word** marks "
                "(upstream behaves the same); audio unchanged")
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
        return (V56Feeds(sequence=seq, speaker_ids=spk, type_ids=ti,
                         base_ids=base, focus_mask=fm, durs_rate=rate),
                stressed)
