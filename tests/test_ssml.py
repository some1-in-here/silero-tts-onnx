"""SSML suite: parser contract (torch-free, no models) + behavior on models.

Parser mirrors upstream v5 process_ssml* (tables, defaults, errors).
Behavior: prosody rate/pitch reshape output, <break> forces exact pause
frames via the engine two-pass path, plain SSML matches plain text.
"""
import os
import sys

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


def _default_models():
    envdir = os.environ.get(env.MODELS_DIR_VAR)
    if envdir:
        return envdir
    local = os.path.normpath(os.path.join(ROOT, "models", "onnx"))
    if not os.path.isfile(os.path.join(local, "acoustic_v5_ru.onnx")):
        raise SystemExit(
            "tests: no ONNX models in " + local + " - build them once: "
            "python scripts/export_all.py (see scripts/README.md), or set "
            + env.MODELS_DIR_VAR + ")")
    return local


MODELS = _default_models()

_tts = None


def tts():
    global _tts
    if _tts is None:
        _tts = SileroTTS(model_dir=MODELS, version="v5_ru")
    return _tts


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


def test_parse_unknown_tag_warns_and_drops_text():
    with pytest.warns(UserWarning):
        segs = parse_ssml('<speak>A. <foo>zzz</foo> B.</speak>')
    joined = " ".join(s["text"] for s in segs)
    assert "zzz" not in joined and "B." in joined


def test_ms_to_frames():
    assert ms_to_frames(500) == 40 and ms_to_frames(150) == 12


def test_examples_ssml_inputs_parse():
    # every SSML payload shown in --examples must parse (examples never lie)
    payloads = [
        "<speak>Сейчас будет пауза. <break time='500ms'/> Вот она была.</speak>",
        "<speak>Сейчас будет пауза. <break time='2s'/> Вот она была.</speak>",
        "<speak>Сейчас будет пауза. <break strength='weak'/> Вот она была.</speak>",
        "<speak>Первая часть. <break strength='x-strong'/> Вторая часть "
        "после очень длинной паузы.</speak>",
        "<speak>Раз. <break time='250ms'/> Два. "
        "<break strength='strong'/> Три.</speak>",
        "<speak><prosody rate='fast'>Я говорю быстро-быстро!</prosody> "
        "<prosody rate='slow'>А теперь медленно и размеренно.</prosody>"
        "А теперь как обычно.</speak>",
        "<speak><prosody pitch='high'>Здесь я говорю высоко.</prosody> "
        "<prosody pitch='low'>А здесь низко.</prosody></speak>",
        "<speak><prosody rate='slow' pitch='low'>Говорю медленно и низко."
        "</prosody></speak>",
        "<speak><prosody rate='slow'>Говорю медленно, "
        "<prosody pitch='high'>а это слово высоко</prosody> и снова ровно."
        "</prosody></speak>",
        "<speak><s>Первое.</s><s>Второе.</s></speak>",
    ]
    for p in payloads:
        segs = parse_ssml(p)
        assert any(s["text"].strip() for s in segs), p


# -- behavior on models ------------------------------------------------

def test_ssml_plain_matches_plain():
    a1, _ = tts().synthesize("Привет! Меня зовут Ксения.", speaker="xenia")
    a2, _ = tts().synthesize("<speak>Привет! Меня зовут Ксения.</speak>",
                             speaker="xenia", ssml=True)
    assert len(a1) == len(a2)
    assert float(np.abs(a1 - a2).max()) == 0.0


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


def test_prosody_pitch_changes_audio():
    a1, _ = tts().synthesize("<speak>Привет.</speak>", speaker="xenia",
                             ssml=True)
    a2, _ = tts().synthesize(
        "<speak><prosody pitch=\"high\">Привет.</prosody></speak>",
        speaker="xenia", ssml=True)
    assert len(a1) == len(a2)
    assert float(np.abs(a1 - a2).max()) > 1e-3


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


def test_ssml_empty_text_fails():
    with pytest.raises(ValueError):
        tts().synthesize("<speak></speak>", speaker="xenia", ssml=True)


def test_ssml_invalid_xml_fails():
    with pytest.raises(ValueError):
        tts().synthesize("Привет <break>", speaker="xenia", ssml=True)


def test_cli_ssml_smoke():
    import tempfile
    from silero_onnx.engine import main
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.wav")
        main(["<speak>Привет.</speak>", "-o", out, "--speaker", "xenia",
              "--ssml", "--model-version", "v5_ru"])
        assert os.path.isfile(out) and os.path.getsize(out) > 10000


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
