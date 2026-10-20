"""v5_6 strategy: emphasis-aware text -> ONNX predictors -> audio.

The strategy contains only the v5_6 session sequence and feed assembly.
SSML parsing/merging/rate + pitch attribution is delegated to the shared
``SsmlLoweringService``; pauses to ``BreakProcessor``; feed validation
to ``FeedBuilder``. Version capabilities come from ``Version_V56``.
"""
import warnings
from collections import namedtuple

import numpy as np

from ...frontend import normalize
from ..base import AbstractTTSStrategy
from ..common.breaks import PauseService
from ..common.feeds import zero_focus_mask
from ...common.ssml_lowering import SsmlLoweringService
from . import emphasis as E

Feeds_V56 = namedtuple("Feeds_V56", ("sequence", "speaker_ids", "type_ids",
                                     "base_ids", "focus_mask", "durs_rate",
                                     "pitch_coefs"))
#: Historical spelling kept for callers that imported the old name.
V56Feeds = Feeds_V56

#: Backward-compatible constant (the value now lives in the ModelSpec).
_PAUSE_SIL_VALUE = -5.0


class Strategy_V56(AbstractTTSStrategy):
    """Torch-free Russian TTS (v5_6 voices, 48 kHz, CPU)."""

    version_id = "v5_6_ru"

    def __init__(self, model_dir, data_dir, sessions, frontend, version=None,
                 language=None):
        super().__init__(model_dir, data_dir, sessions, frontend,
                         version=version, language=language)
        self._breaks = PauseService(self._version.pause_sil_value)
        self._ssml = SsmlLoweringService(
            normalize=self._normalize,
            supports_pitch=self._version.supports_pitch,
            version_id=self._version.version_id,
            segment_transform=E.strip_markup_tags)
        # Speaker-table validation is structural, not nominal: ids must
        # be dense 0..N-1 with unique non-empty names (no hardcoded
        # voice list anywhere — a new upstream voice flows through with
        # zero code changes once models + speakers.json are refreshed).
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

    def _synthesize(self, text, speaker, ssml=False):
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
        feeds = self._feeds.acoustic_feeds(
            sequence=seq, speaker_ids=spk, durs_rate=f.durs_rate,
            gt_durs=gt_d, gt_pitch=pit.astype(np.float32),
            pitch_coefs=f.pitch_coefs, type_ids=ti, focus_mask=fm)
        acoustic = self._sessions["acoustic"]
        if breaks:
            mel, dur = self._breaks.synthesize(acoustic, feeds, breaks)
        else:
            mel, dur = acoustic.run(None, feeds)
        audio = self._sessions["vocoder"].run(
            None, {"mel": mel.astype(np.float32)})[0]
        return np.ascontiguousarray(audio.squeeze()), dur

    def _prepare(self, text, speaker):
        """Text -> Feeds_V56 namedtuple (numpy in / numpy out, no sessions)."""
        return self._build(text, speaker)[0]

    def _prepare_ssml(self, text, speaker):
        """<speak> XML -> (Feeds_V56, breaks); rate + pause plans."""
        if not self._version.supports_ssml:
            raise ValueError(
                f"SSML is not supported for version "
                f"'{self._version.version_id}'")
        intensity_tag = '<intensity' in text
        if intensity_tag:
            text, intensity = E.extract_intensity(text)
        else:
            intensity = None
        plan = self._ssml.plan(text)
        if not plan.segments:
            raise ValueError(
                "silero-onnx: error: SSML holds no speakable text")
        feeds, stressed = self._build(
            " ".join(s.text for s in plan.segments), speaker,
            intensity=intensity, intensity_tag=intensity_tag)
        rates, breaks = self._ssml.attribute(
            stressed, feeds.sequence[0], self._frontend.symbol_to_id,
            plan.segments, plan.leading_break_ms)
        pitches = self._ssml.attribute_pitch(
            stressed, feeds.sequence[0], self._frontend.symbol_to_id,
            plan.segments, plan.leading_break_ms)
        feeds = feeds._replace(
            durs_rate=np.array([rates], dtype=np.float32),
            pitch_coefs=np.array([pitches], dtype=np.float32))
        return feeds, breaks

    def _build(self, text, speaker, intensity=None, intensity_tag=None):
        """Marked text -> (Feeds_V56, stressed); shared plain/SSML path."""
        if not self._version.supports_stress:
            if "*" in text:
                raise ValueError(
                    f"emphasis markers are not supported for version "
                    f"'{self._version.version_id}'")
            if "<intensity" in text:
                warnings.warn(
                    f"silero-onnx: {self._version.version_id} ignores "
                    "<intensity=...>: this version has no focus channel")
            intensity_tag = False
            intensity = None
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
        if has_focus and self._version.supports_stress:
            accented = E.apply_focus_marks(stressed, norm_marked)
            _c, fm = E.parse_focus_from_text(accented, n, table,
                                             intensity=intensity)
        else:
            fm = zero_focus_mask(n)
        spans, sentence_texts = E.token_sent_spans(ids, self._id_to_symbol)
        types = E.classify_sentence_types(
            clean,
            classify_questions=self._version.supports_questions,
        ).split("|")
        if len(types) != len(spans):
            types = [
                E.classify_sentence_type(
                    sentence.replace("+", ""),
                    classify_questions=self._version.supports_questions,
                )
                for sentence in sentence_texts
            ]
        emph_sentences = E.emph_spans_from_focus(
            spans, fm[0].tolist(), emph_idx)
        ti = E.build_sentence_type_ids(
            ids, types, emph_sentences, self._id_to_symbol)
        base = (ti % len(E.SENTENCE_TYPE_TO_ID)).astype(np.int64)
        rate = np.ones((1, n), dtype=np.float32)
        pitch = np.ones((1, n), dtype=np.float32)
        return (Feeds_V56(sequence=seq, speaker_ids=spk, type_ids=ti,
                         base_ids=base, focus_mask=fm, durs_rate=rate,
                         pitch_coefs=pitch),
                stressed)


#: Historical class name (v5_6 spelling used by older imports/tests).
Strategy_V5_6 = Strategy_V56

__all__ = ["Strategy_V56", "Strategy_V5_6", "Feeds_V56", "V56Feeds"]
