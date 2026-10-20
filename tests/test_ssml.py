"""SSML suite: parser contract + lowering (torch-free) + behavior on models.

Parser mirrors upstream v5 process_ssml* (tables, defaults, errors).
v5_ru behavior: prosody rate/pitch reshape output, <break> forces exact
pause frames via the two-pass path, plain SSML matches plain text.
v5_6_ru: the same rate/pitch/break lowering is pinned torch-free on a
stub frontend (strategy._prepare_ssml feeds) and measured on the real
models; <prosody pitch> reaches its acoustic pitch_coefs input.
"""
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

assert "torch" not in sys.modules, "test process imported torch before package!"

from silero_onnx import env  # noqa: E402

from silero_onnx import SileroTTS  # noqa: E402
from silero_onnx.ssml import (  # noqa: E402
    ms_to_frames,
    parse_ssml,
)
from silero_onnx.strategies.v5_6 import Strategy_V5_6  # noqa: E402


def _default_models():
    """Model dir: SILERO_ONNX_MODELS override, else the local build
    output (<repo>/models/onnx). Behavior tests are marked and skipped
    with a reason when the set is absent (see tests/conftest.py);
    parser tests above never touch models."""
    envdir = os.environ.get(env.MODELS_DIR_VAR)
    if envdir:
        return envdir
    return os.path.normpath(os.path.join(ROOT, "models", "onnx"))


MODELS = _default_models()

_tts = None


def tts():
    global _tts
    if _tts is None:
        _tts = SileroTTS(model_dir=MODELS, version="v5_ru")
    return _tts


_tts56 = None


def tts56():
    global _tts56
    if _tts56 is None:
        _tts56 = SileroTTS(model_dir=MODELS, version="v5_6_ru")
    return _tts56


# -- v5_6 lowering (torch-free: pure text + numpy, stub frontend) ------

class _V56Frontend:
    """Stand-in for TorchFreeFrontend (same seam as test_factory)."""

    speakers = {"aidar": 0, "baya": 1, "kseniya": 2, "eugene": 3,
                "xenia": 4}

    @property
    def symbol_to_id(self):
        from silero_onnx.frontend import load_symbol_table
        return load_symbol_table()[1]

    def text_to_ids(self, text):
        from silero_onnx.frontend import add_sos_eos, normalize, tokenize
        t = normalize(text)
        table = self.symbol_to_id
        return add_sos_eos(tokenize(t, table, errors="filter"), table), t


_v56_strategy = None


def v56_strategy():
    """v5_6 strategy on a stub frontend (no models, no neural stress).

    Without the neural stress there are no '+' marks, so the token
    positions below are plain char walks: deterministic and model-free.
    """
    global _v56_strategy
    if _v56_strategy is None:
        _v56_strategy = Strategy_V5_6("m", "d", {}, _V56Frontend())
    return _v56_strategy


def test_v56_ssml_plain_lowers_to_plain_feeds():
    st = v56_strategy()
    f0 = st._prepare("Привет, мир.", "xenia")
    f1, breaks = st._prepare_ssml("<speak>Привет, мир.</speak>", "xenia")
    assert breaks == []
    assert (f0.sequence == f1.sequence).all()
    assert (f0.type_ids == f1.type_ids).all()
    assert (f0.focus_mask == f1.focus_mask).all()
    assert f1.durs_rate.shape == f1.sequence.shape
    assert bool((f1.durs_rate == 1.0).all())


def test_v56_ssml_rate_holds_on_segment_words():
    st = v56_strategy()
    f, breaks = st._prepare_ssml(
        "<speak>раз <prosody rate='fast'>два</prosody> три.</speak>",
        "xenia")
    marks = "".join(st._id_to_symbol[i] for i in f.sequence[0])
    rates = f.durs_rate[0]
    i = marks.index("два")
    assert breaks == []
    # the segment's chars carry its rate; the space on its left keeps the
    # LEFT segment's rate (v5 join-space convention), the one on its right
    # belongs to the segment it closes; sos/eos take the edge segments
    assert float(rates[i - 1]) == 1.0
    assert list(rates[i:i + 3]) == pytest.approx([1.2, 1.2, 1.2])
    assert float(rates[i + 3]) == pytest.approx(1.2)
    assert float(rates[0]) == 1.0 and float(rates[-1]) == 1.0
    changed = np.nonzero(~np.isclose(rates, 1.0))[0]
    assert changed.size == 4


def test_v56_ssml_break_plan_lands_on_segment_final_token():
    st = v56_strategy()
    f, breaks = st._prepare_ssml(
        '<speak>раз. <break time="500ms"/> два.</speak>', "xenia")
    marks = "".join(st._id_to_symbol[i] for i in f.sequence[0])
    assert breaks == [(marks.index("."), ms_to_frames(500))]
    assert float(f.durs_rate[0].max()) == 1.0  # breaks never bend rates


def test_v56_ssml_p_s_imply_break_plans():
    st = v56_strategy()
    f, breaks = st._prepare_ssml("<speak><s>раз.</s><s>два.</s></speak>",
                                 "xenia")
    marks = "".join(st._id_to_symbol[i] for i in f.sequence[0])
    assert [p for p, _ in breaks] == [marks.index("."), marks.rindex(".")]
    assert [n for _, n in breaks] == [ms_to_frames(300)] * 2


def test_v56_ssml_pitch_lands_in_feeds():
    st = v56_strategy()
    plain, _ = st._prepare_ssml("<speak>раз два.</speak>", "xenia")
    f, breaks = st._prepare_ssml(
        "<speak><prosody pitch='high'>раз два.</prosody></speak>",
        "xenia")
    assert (f.sequence == plain.sequence).all()
    assert bool((f.durs_rate == 1.0).all()) and breaks == []
    assert f.pitch_coefs.shape == f.sequence.shape
    assert np.allclose(f.pitch_coefs, 1.15)

    # <prosody pitch="medium"> is an explicit unit coefficient: same
    # feeds and no warning (upstream's 1.0 is a no-op).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m, _ = st._prepare_ssml(
            "<speak><prosody pitch='medium'>раз два.</prosody></speak>",
            "xenia")
        r, _ = st._prepare_ssml(
            "<speak><prosody pitch='robot'>раз два.</prosody></speak>",
            "xenia")
    assert np.array_equal(m.pitch_coefs, plain.pitch_coefs)
    assert np.array_equal(r.pitch_coefs, np.zeros_like(r.pitch_coefs))


def test_v56_ssml_markers_inside_segments():
    st = v56_strategy()
    f, breaks = st._prepare_ssml("<speak>Меня зовут **Ксения**.</speak>",
                                 "xenia")
    assert breaks == []
    assert set(np.unique(f.focus_mask).tolist()) == {0, 3}
    assert bool((f.durs_rate == 1.0).all())


def test_v56_intensity_scales_marks_plain_and_ssml():
    st = v56_strategy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # marks present: no no-op warning
        fp = st._prepare("<intensity=2>*раз* два.", "xenia")
        fe = st._prepare("<intensity=1>**раз** два.", "xenia")
        fs, breaks = st._prepare_ssml(
            "<speak><intensity=1>*раз* два.</speak>", "xenia")
    # the level lands in the focus mask; ** scales exactly like *
    assert set(np.unique(fp.focus_mask).tolist()) == {0, 2}
    assert set(np.unique(fe.focus_mask).tolist()) == {0, 1}
    assert set(np.unique(fs.focus_mask).tolist()) == {0, 1}
    assert breaks == []


def test_v56_intensity_without_marks_warns_and_is_inert():
    st = v56_strategy()
    with pytest.warns(UserWarning, match="intensity"):
        fp = st._prepare("<intensity=1>раз два.</intensity>", "xenia")
    with pytest.warns(UserWarning, match="intensity"):
        fs, breaks = st._prepare_ssml(
            "<speak><intensity=1>раз два.</intensity></speak>", "xenia")
    assert not fp.focus_mask.any() and not fs.focus_mask.any()
    assert breaks == []


def test_v56_multiword_star_span_marks_each_word():
    # *...* may span several words: upstream re-marks each word of the
    # span individually (span == explicit per-word marks; bit-identical
    # audio verified against the torch model), and the feeds agree
    st = v56_strategy()
    span = st._prepare("*Тихо говорю.* *Громко говорю.* Нормально говорю.",
                       "xenia")
    expl = st._prepare(
        "*Тихо* *говорю.* *Громко* *говорю.* Нормально говорю.", "xenia")
    assert (span.focus_mask == expl.focus_mask).all()
    assert (span.sequence == expl.sequence).all()
    assert (span.type_ids == expl.type_ids).all()


def test_v56_user_report_intensity_per_utterance_and_word_emphasis():
    # user-reported pair: #1 works (multi-word * spans are legal, but
    # the FIRST <intensity=N> wins for the whole utterance -> both
    # phrases at level 1; differing values warn); #2 fails with
    # upstream's own emphasis error (** wraps a single word only)
    st = v56_strategy()
    with pytest.warns(UserWarning, match="first one"):
        f = st._prepare(
            "<intensity=1>*Тихо говорю.*</intensity> "
            "<intensity=3>*Громко говорю.*</intensity> Нормально говорю.",
            "xenia")
    assert set(np.unique(f.focus_mask).tolist()) == {0, 1}
    # the multi-value warning fires before the markup error; catch it so
    # pytest's summary stays clean and pin it too
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="эмфазы"):
            st._prepare(
                "<intensity=1>**Тихо говорю.**</intensity> "
                "<intensity=3>**Громко говорю.**</intensity> "
                "Нормально говорю.",
                "xenia")
    assert any("first one" in str(w.message) for w in rec)


def test_v56_ssml_empty_fails_loud():
    with pytest.raises(ValueError, match="speakable"):
        v56_strategy()._prepare_ssml("<speak></speak>", "xenia")


def test_v56_ssml_arity_guard_fails_loud():
    class Wobbly(_V56Frontend):
        def text_to_ids(self, text):
            ids, t = super().text_to_ids(text)
            return ids, t.replace(" ", "", 1)  # "stress" merged two words

    st = Strategy_V5_6("m", "d", {}, Wobbly())
    with pytest.raises(ValueError, match="arity"):
        st._prepare_ssml("<speak>раз два.</speak>", "xenia")


# -- parser: structure ------------------------------------------------

def test_parse_basic_break():
    segs = parse_ssml('<speak>Привет. <break time="500ms"/> Пока.</speak>')
    assert len(segs) == 2
    assert segs[0]["break_ms"] == 500 and segs[0]["rate"] == 1.0
    assert segs[1]["break_ms"] is None and segs[1]["pitch"] == 1.0


def test_parse_missing_speak():
    with pytest.raises(ValueError):
        parse_ssml('Привет. <break time="500ms"/> Пока.')


def test_parse_bad_xml():
    with pytest.raises(ValueError):
        parse_ssml('<speak>Привет.')


def test_parse_trailing_text_hint():
    with pytest.raises(ValueError) as e:
        parse_ssml('</speak> лишнее')
    assert 'speak' in str(e.value).lower()


def test_parse_strength_only():
    segs = parse_ssml('<speak>A. <break strength="weak"/> B.</speak>')
    assert segs[0]["break_ms"] == 75


def test_parse_bad_strength():
    with pytest.raises(ValueError):
        parse_ssml('<speak>A. <break strength="mega"/> B.</speak>')


def test_parse_bad_time_unit():
    with pytest.raises(ValueError):
        parse_ssml('<speak>A. <break time="2x"/> B.</speak>')


def test_parse_fractional_seconds_rejected():
    # upstream int() parsing rejects these too — parity, pinned
    with pytest.raises(ValueError):
        parse_ssml('<speak>A. <break time="1.5s"/> B.</speak>')


def test_parse_clamp_5s():
    with pytest.warns(UserWarning):
        segs = parse_ssml('<speak>A. <break time="9000ms"/> B.</speak>')
    assert segs[0]["break_ms"] == 5000


def test_parse_prosody_values():
    segs = parse_ssml('<speak><prosody rate="fast" pitch="high">A.</prosody>'
                      ' B.</speak>')
    assert segs[0]["rate"] == 1.2 and segs[0]["pitch"] == 1.15
    assert segs[1]["rate"] == 1.0 and segs[1]["pitch"] == 1.0


def test_parse_prosody_percent():
    segs = parse_ssml('<speak><prosody rate="150%">A.</prosody> '
                      '<prosody pitch="+12%">B.</prosody> '
                      '<prosody pitch="-10%">C.</prosody></speak>')
    assert segs[0]["rate"] == 1.5
    assert segs[1]["pitch"] == pytest.approx(1.12)
    assert segs[2]["pitch"] == pytest.approx(0.9)


def test_parse_empty_prosody():
    with pytest.raises(ValueError):
        parse_ssml('<speak><prosody>A.</prosody></speak>')


def test_parse_bad_rate():
    with pytest.raises(ValueError):
        parse_ssml('<speak><prosody rate="ludicrous">A.</prosody></speak>')


def test_parse_p_s_imply_breaks():
    segs = parse_ssml('<speak><s>A.</s><s>B.</s></speak>')
    assert segs[0]["break_ms"] == 300 and segs[1]["break_ms"] == 300
    segs = parse_ssml('<speak><p>A.</p></speak>')
    assert segs[0]["break_ms"] == 1000


def test_parse_unknown_tag_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported SSML tag"):
        parse_ssml('<speak>A. <foo>zzz</foo> B.</speak>')


def test_parse_unknown_attributes_raise_value_error():
    with pytest.raises(ValueError, match="Unsupported SSML <break> attribute"):
        parse_ssml('<speak>A. <break time="1s" foo="x"/> B.</speak>')
    with pytest.raises(ValueError,
                       match="Unsupported SSML <prosody> attribute"):
        parse_ssml('<speak><prosody rate="slow" foo="x">A.</prosody></speak>')


def test_ms_to_frames():
    assert ms_to_frames(500) == 40 and ms_to_frames(150) == 12


def test_examples_ssml_inputs_parse():
    # every SSML payload shown in --examples must parse (examples never
    # lie); payloads are extracted from the printed examples themselves,
    # so the check cannot drift from what users see
    import re
    from silero_onnx.engine import examples_text
    text = re.sub(r"\033\[[0-9;]*m", "", examples_text())
    payloads = re.findall(r'--ssml "([^"]+)"', text)
    assert len(payloads) >= 8  # guard: extraction found the examples
    for p in payloads:
        segs = parse_ssml(p)
        assert any(s["text"].strip() for s in segs), p


# -- behavior on models ------------------------------------------------

@pytest.mark.requires_models
def test_ssml_plain_matches_plain():
    a1, _ = tts().synthesize("Привет! Меня зовут Ксения.", speaker="xenia")
    a2, _ = tts().synthesize("<speak>Привет! Меня зовут Ксения.</speak>",
                             speaker="xenia", ssml=True)
    assert len(a1) == len(a2)
    assert float(np.abs(a1 - a2).max()) == 0.0


@pytest.mark.requires_models
def test_prosody_rate_shortens():
    text = "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"
    a_slow, d_slow = tts().synthesize(
        f"<speak><prosody rate=\"x-slow\">{text}</prosody></speak>",
        speaker="xenia", ssml=True)
    a_fast, d_fast = tts().synthesize(
        f"<speak><prosody rate=\"x-fast\">{text}</prosody></speak>",
        speaker="xenia", ssml=True)
    # rate scales per-token frames: visible in dur sums and audio length
    # (dur.shape[1] is the token count — identical by construction)
    assert float(d_slow.sum()) > float(d_fast.sum())
    assert len(a_fast) < len(a_slow)


@pytest.mark.requires_models
def test_prosody_pitch_changes_audio():
    a1, _ = tts().synthesize("<speak>Привет.</speak>", speaker="xenia",
                             ssml=True)
    a2, _ = tts().synthesize(
        "<speak><prosody pitch=\"high\">Привет.</prosody></speak>",
        speaker="xenia", ssml=True)
    assert len(a1) == len(a2)
    assert float(np.abs(a1 - a2).max()) > 1e-3


@pytest.mark.requires_models
def test_break_forces_exact_frames():
    a0, d0 = tts().synthesize(
        "<speak>Подожди. Продолжаю.</speak>", speaker="xenia", ssml=True)
    a1, d1 = tts().synthesize(
        '<speak>Подожди. <break time="500ms"/> Продолжаю.</speak>',
        speaker="xenia", ssml=True)
    # '.' predicts 24 frames on its own; the break forces exactly 40 and
    # touches nothing else (all other positions bit-identical)
    assert bool((d1[0] == 40).any()), "500ms break must force a 40-frame phone"
    assert float(d1.sum() - d0.sum()) == 16.0
    assert len(a1) - len(a0) == 9600


@pytest.mark.requires_models
def test_break_on_comma_behaves():
    # comma breaks skip upstream's batch-2 support re-prediction
    # (documented approximation): pause position/length still exact
    a0, d0 = tts().synthesize(
        "<speak>Подожди, продолжаю.</speak>", speaker="xenia", ssml=True)
    a1, d1 = tts().synthesize(
        '<speak>Подожди, <break time="300ms"/> продолжаю.</speak>',
        speaker="xenia", ssml=True)
    # ',' predicts 8 frames on its own; the break forces exactly 24 and
    # touches nothing else
    assert bool((d1[0] == 24).any()), "300ms break must force a 24-frame phone"
    assert float(d1.sum() - d0.sum()) == 16.0
    assert len(a1) - len(a0) == 9600


@pytest.mark.requires_models
def test_ssml_empty_text_fails():
    with pytest.raises(ValueError):
        tts().synthesize("<speak></speak>", speaker="xenia", ssml=True)


@pytest.mark.requires_models
def test_ssml_invalid_xml_fails():
    with pytest.raises(ValueError):
        tts().synthesize("Привет <break>", speaker="xenia", ssml=True)


@pytest.mark.requires_models
def test_cli_ssml_smoke():
    import tempfile
    from silero_onnx.engine import main
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.wav")
        main(["<speak>Привет.</speak>", "-o", out, "--speaker", "xenia",
              "--ssml", "--model-version", "v5_ru"])
        assert os.path.isfile(out) and os.path.getsize(out) > 10000


@pytest.mark.requires_models
def test_cli_invalid_ssml_clean_error():
    # CLI converts ValueError to a one-line SystemExit — no traceback
    import tempfile
    from silero_onnx.engine import main
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.wav")
        with pytest.raises(SystemExit) as e:
            main(["<speak>A.</speak> tail", "-o", out, "--ssml",
                  "--model-version", "v5_ru"])
        assert "speak" in str(e.value.code).lower()
        assert not os.path.exists(out)


# -- v5_6 behavior on models -------------------------------------------

@pytest.mark.requires_v56
def test_v56_ssml_plain_matches_plain():
    a1, d1 = tts56().synthesize("Привет! Меня зовут Ксения.",
                                speaker="xenia")
    a2, d2 = tts56().synthesize("<speak>Привет! Меня зовут Ксения.</speak>",
                                speaker="xenia", ssml=True)
    assert len(a1) == len(a2)
    assert float(np.abs(a1 - a2).max()) == 0.0
    assert bool((d1 == d2).all())


@pytest.mark.requires_v56
def test_v56_prosody_rate_changes_length():
    text = "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"
    a_slow, d_slow = tts56().synthesize(
        f"<speak><prosody rate=\"x-slow\">{text}</prosody></speak>",
        speaker="xenia", ssml=True)
    a_fast, d_fast = tts56().synthesize(
        f"<speak><prosody rate=\"x-fast\">{text}</prosody></speak>",
        speaker="xenia", ssml=True)
    # rate scales per-token frames: visible in dur sums and audio length
    # (dur.shape[1] is the token count — identical by construction)
    assert d_slow.shape == d_fast.shape
    assert float(d_slow.sum()) > float(d_fast.sum())
    assert len(a_fast) < len(a_slow)


@pytest.mark.requires_v56
def test_v56_prosody_pitch_changes_pitch_channel():
    text = "Привет! Меня зовут Ксения."
    a1, _ = tts56().synthesize(text, speaker="xenia")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a2, _ = tts56().synthesize(
            f"<speak><prosody pitch='high'>{text}</prosody></speak>",
            speaker="xenia", ssml=True)
        a3, _ = tts56().synthesize(
            f"<speak><prosody pitch='medium'>{text}</prosody></speak>",
            speaker="xenia", ssml=True)
        a4, _ = tts56().synthesize(
            f"<speak><prosody pitch='robot'>{text}</prosody></speak>",
            speaker="xenia", ssml=True)
    assert len(a1) == len(a2) == len(a4)
    assert float(np.abs(a1 - a2).max()) > 1e-3
    assert float(np.abs(a1 - a4).max()) > 1e-3
    # medium is the upstream unit coefficient; it must be a true no-op
    assert float(np.abs(a1 - a3).max()) == 0.0


@pytest.mark.requires_v56
def test_v56_break_forces_exact_frames():
    a0, d0 = tts56().synthesize("<speak>Подожди. Продолжаю.</speak>",
                                speaker="xenia", ssml=True)
    a1, d1 = tts56().synthesize(
        '<speak>Подожди. <break time="500ms"/> Продолжаю.</speak>',
        speaker="xenia", ssml=True)
    diff = d1[0] - d0[0]
    changed = np.nonzero(diff)[0]
    # the '.' token is forced to exactly 40 frames and nothing else moves
    assert changed.size == 1, "the break must touch exactly one token"
    assert float(d1[0, changed[0]]) == 40.0
    assert float(diff.sum()) > 0.0
    # audio grows by exactly the duration delta (12.5ms mel hop @ 48kHz)
    assert len(a1) - len(a0) == int(float(diff.sum())) * 600


@pytest.mark.requires_v56
def test_v56_break_on_comma_behaves():
    a0, d0 = tts56().synthesize("<speak>Подожди, продолжаю.</speak>",
                                speaker="xenia", ssml=True)
    a1, d1 = tts56().synthesize(
        '<speak>Подожди, <break time="300ms"/> продолжаю.</speak>',
        speaker="xenia", ssml=True)
    diff = d1[0] - d0[0]
    changed = np.nonzero(diff)[0]
    assert changed.size == 1
    assert float(d1[0, changed[0]]) == 24.0
    assert len(a1) - len(a0) == int(float(diff.sum())) * 600


@pytest.mark.requires_v56
def test_v56_break_at_tail_capped_like_upstream():
    # '.' sits at position -2, which the graph pins to 13 frames; upstream
    # feeds its symb_durs override through the same tail caps, so a
    # 500ms request cannot win there — durations stay identical and the
    # audio keeps its length (fx_pauses still flattens the mel region)
    a0, d0 = tts56().synthesize("<speak>Привет.</speak>", speaker="xenia",
                                ssml=True)
    a1, d1 = tts56().synthesize(
        '<speak>Привет. <break time="500ms"/></speak>', speaker="xenia",
        ssml=True)
    assert bool((d1 == d0).all())
    assert len(a1) == len(a0)
    assert float(np.abs(a1 - a0).max()) > 0.0


@pytest.mark.requires_v56
def test_v56_intensity_scales_focus_only():
    t = tts56()
    # no * focus marks: the tag is inert (upstream audio is identical)
    with pytest.warns(UserWarning, match="intensity"):
        q1, _ = t.synthesize("<intensity=1>Тихо говорю.</intensity>",
                             speaker="xenia")
        q3, _ = t.synthesize("<intensity=3>Тихо говорю.</intensity>",
                             speaker="xenia")
    assert len(q1) == len(q3)
    assert float(np.abs(q1 - q3).max()) == 0.0
    # with a *focus* mark the level rescales the speech (audio + length)
    f1, _ = t.synthesize("<intensity=1>*Тихо* говорю.</intensity>",
                         speaker="xenia")
    f3, _ = t.synthesize("<intensity=3>*Тихо* говорю.</intensity>",
                         speaker="xenia")
    assert not (len(f1) == len(f3) and np.array_equal(f1, f3))
    # SSML strips the pseudo-tag before XML parsing (upstream order):
    # the same utterance lowers to bit-identical audio
    p2, _ = t.synthesize("<intensity=2>*Тихо* говорю.</intensity>",
                         speaker="xenia")
    s2, _ = t.synthesize(
        "<speak><intensity=2>*Тихо* говорю.</intensity></speak>",
        speaker="xenia", ssml=True)
    assert len(p2) == len(s2)
    assert float(np.abs(p2 - s2).max()) == 0.0


@pytest.mark.requires_v56
def test_v56_ssml_empty_fails():
    with pytest.raises(ValueError):
        tts56().synthesize("<speak></speak>", speaker="xenia", ssml=True)


@pytest.mark.requires_v56
def test_v56_ssml_invalid_xml_fails():
    with pytest.raises(ValueError):
        tts56().synthesize("Привет <break>", speaker="xenia", ssml=True)


@pytest.mark.requires_v56
def test_v56_cli_ssml_smoke():
    import tempfile
    from silero_onnx.engine import main
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.wav")
        main(["<speak>Привет.</speak>", "-o", out, "--speaker", "xenia",
              "--ssml", "--model-version", "v5_6_ru"])
        assert os.path.isfile(out) and os.path.getsize(out) > 10000


@pytest.mark.requires_v56
def test_v56_cli_invalid_ssml_clean_error():
    import tempfile
    from silero_onnx.engine import main
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.wav")
        with pytest.raises(SystemExit) as e:
            main(["<speak>A.</speak> tail", "-o", out, "--ssml",
                  "--model-version", "v5_6_ru"])
        assert "speak" in str(e.value.code).lower()
        assert not os.path.exists(out)


def test_cli_examples_runs_without_models():
    # --examples exits before SileroTTS init: no models needed
    import io
    from contextlib import redirect_stdout
    from silero_onnx.engine import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["--examples"]) is None
    out = buf.getvalue()
    for section in ("Basic", "Voices", "Stress marks", "Punctuation",
                    "SSML", "Custom models"):
        assert section in out


def test_cli_examples_are_self_describing():
    # policy (engine.examples_text docstring): no trailing '# ...'
    # legends — a copied command is understood on its own; every
    # non-list example quotes text and writes a .wav
    from silero_onnx.engine import examples_text
    text = examples_text()
    assert "#" not in text
    lines = [ln for ln in text.splitlines()
             if ln.strip().startswith("silero-onnx")]
    assert lines
    for ln in lines:
        if "--model-list" in ln or "--speaker-list" in ln:
            continue
        assert " -o " in ln and ".wav" in ln and '"' in ln, ln


def test_cli_examples_plain_without_tty():
    # NO_COLOR or pipes: zero ANSI escapes in output
    import io
    import os as _os
    from contextlib import redirect_stdout
    from silero_onnx.engine import examples_text
    old = _os.environ.get(env.NO_COLOR_VAR)
    _os.environ[env.NO_COLOR_VAR] = "1"
    try:
        buf = io.StringIO()  # not a tty either
        with redirect_stdout(buf):
            text = examples_text()
        assert "\033[" not in text and "\033[" not in buf.getvalue()
    finally:
        if old is None:
            del _os.environ[env.NO_COLOR_VAR]
        else:
            _os.environ[env.NO_COLOR_VAR] = old


def test_cli_model_list_needs_no_models(monkeypatch):
    # --model-list prints versions without touching model files: any
    # SileroTTS construction attempt fails the test outright.
    import io
    from contextlib import redirect_stdout
    import silero_onnx.engine as _eng

    def _bomb(*a, **k):
        raise AssertionError("SileroTTS must not be constructed here")

    monkeypatch.setattr(_eng, "SileroTTS", _bomb)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert _eng.main(["--model-list"]) is None
    assert buf.getvalue().split() == ["v5_6_ru", "v5_ru"]


def test_cli_speaker_list_needs_no_models(monkeypatch):
    # --speaker-list works on a bare checkout: this is the regression
    # test for constructing SileroTTS (and loading all models) before
    # handling the flag.
    import io
    from contextlib import redirect_stdout
    import silero_onnx.engine as _eng

    def _bomb(*a, **k):
        raise AssertionError("SileroTTS must not be constructed here")

    monkeypatch.setattr(_eng, "SileroTTS", _bomb)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert _eng.main(["--speaker-list"]) is None
    assert buf.getvalue().split() == ["aidar", "baya", "eugene",
                                      "kseniya", "xenia"]


def test_zzz_no_torch_at_end():
    assert "torch" not in sys.modules


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)
           and k != "test_zzz_no_torch_at_end"]
    fns.append(("test_zzz_no_torch_at_end", test_zzz_no_torch_at_end))
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
