"""Production test suite for silero_onnx (torch-free).

Covers: normalization, token mapping, stress modules (homograph + ngram),
SileroTTS facade, speakers list, end-to-end synthesis, error handling.
Uses ONLY the package + stdlib/numpy/soundfile/onnxruntime — never torch
and never build-time helpers (asserted explicitly).

Run:  py -3.12 -m pytest tests/test_production.py -v
  or: py -3.12 tests/test_production.py
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
from silero_onnx import frontend as F  # noqa: E402
from silero_onnx.homograph_solver import FullStress  # noqa: E402
from silero_onnx.ngram_accentor import OnnxNgram  # noqa: E402

def _default_models():
    """Model dir for tests: SILERO_ONNX_MODELS override, else the local
    build output (<repo>/models/onnx from scripts/export_all.py).

    Resolution only: whether a test can actually run against it is
    decided by the markers (tests/conftest.py) — model tests skip with
    an explicit reason when the needed set is absent, so a bare `pytest`
    works on a fresh checkout and in the regular CI workflow."""
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


def test_no_torch_at_import():
    assert "torch" not in sys.modules


def test_runtime_package_has_no_torch_imports():
    """Static AST scan: no `import torch` anywhere under silero_onnx/.

    Runtime behavior is already pinned by the import assertions around
    this suite; this scan enforces the textual guarantee too, so a dead
    build-time helper cannot quietly smuggle torch into the wheel.
    Torch-side recipes (e.g. the symbol/speaker table dump) live in
    scripts/, where torch is allowed.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(F.__file__).parent
    offenders = []
    for py in sorted(pkg.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "torch" for n in names):
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}")
    assert not offenders, f"torch imported under silero_onnx/: {offenders}"


def test_abbreviations():
    assert F.expand_abbreviations("Встречаемся на ул. Ленина, д. 5.") == \
        "Встречаемся на улица Ленина, дом 5."
    assert F.expand_abbreviations("Он родился в г. Москва в 1990 г.") == \
        "Он родился в город Москва в 1990 год"
    assert F.expand_abbreviations("См. стр. 5.") == "Смотри страница 5."
    assert F.expand_abbreviations("До м. Тверская 5 м.") == \
        "До метро Тверская 5 метров"


def test_numbers():
    assert F.expand_numbers("Мне 21 год.") == "Мне двадцать один год."
    assert "тридцать шесть" in F.expand_numbers("Температура 36,6.")
    # digits have no TTS symbols: expansion is required, never passthrough
    assert not any(ch.isdigit() for ch in F.expand_numbers("Было 100 рублей."))


def test_punctuation_kept():
    out = F.expand_abbreviations("Привет, мир! Как дела?")
    assert out == "Привет, мир! Как дела?"


def test_token_mapping():
    _, table = F.load_symbol_table()
    assert len(table) == 47
    assert F.tokenize("привет мир", table) == [26, 27, 19, 13, 16, 29, 46, 23, 19, 27]
    ids = F.add_sos_eos(F.tokenize("привет мир", table), table)
    assert ids[:3] == [2, 26, 27] and ids[-2:] == [27, 1]  # |...~
    try:
        F.tokenize("hello", table)
    except ValueError as e:
        assert "unknown symbols" in str(e)
    else:
        raise AssertionError("latin must raise with errors='raise'")
    assert F.tokenize("hello мир", table, errors="filter") == [46, 23, 19, 27]


@pytest.mark.requires_models
def test_full_stress_exact():
    fs = FullStress(models_dir=MODELS)
    assert fs("На горе стоит старый замок.") == "Н+а гор+е сто+ит ст+арый зам+ок."
    # 'мел/мёл' sits exactly on the ORT decision boundary (yo-model
    # sigmoid ≈ 0.5): observed as М+ел in one run and М+ёл in another.
    # Both are legitimate outputs; pin the set, not one variant.
    out = fs("Мел рассыпался по полу.")
    assert out in ("М+ел расс+ыпался п+о п+олу.",
                   "М+ёл расс+ыпался п+о п+олу."), out


@pytest.mark.requires_models
def test_ngram_structure():
    ng = OnnxNgram(models_dir=MODELS)
    out = ng("Привет! Меня зовут Ксения.")
    assert "+" in out and " " in out
    # re-stressing the de-stressed output still marks stresses
    assert "+" in ng(out.replace("+", ""))


@pytest.mark.requires_models
def test_speakers():
    assert tts().speakers == ["aidar", "baya", "eugene", "kseniya", "xenia"]


@pytest.mark.requires_models
def test_synthesize_basic():
    audio, dur = tts().synthesize("Привет! Меня зовут Ксения.", speaker="xenia")
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) > 48000  # > 1 s of speech
    assert np.all(np.isfinite(audio)) and float(np.abs(audio).max()) > 1e-3
    assert 1.5 < len(audio) / 48000 < 2.5
    # durations: exact T=32 expected, ±1 tolerated (ORT jitter can flip
    # round() at exact .5 boundaries on other inputs)
    assert dur.ndim == 2 and dur.shape[0] == 1 and 30 <= dur.shape[1] <= 34


@pytest.mark.requires_models
def test_synthesize_to_file():
    import soundfile as sf
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.wav")
        tts().synthesize_to_file("Замок был открыт.", out, speaker="xenia")
        audio, sr = sf.read(out)
    assert sr == 48000 and len(audio) > 48000 and np.all(np.isfinite(audio))


@pytest.mark.requires_v56
def test_v56_synthesize_basic():
    """v5_6_ru (the default version) must be covered by the suite too:
    markers -> predictors -> acoustic -> vocoder end to end."""
    tts56 = SileroTTS(model_dir=MODELS, version="v5_6_ru")
    audio, dur = tts56.synthesize("Привет! Меня зовут **Ксения**.",
                                  speaker="xenia")
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) > 24000
    assert np.all(np.isfinite(audio)) and float(np.abs(audio).max()) > 1e-3
    assert dur.ndim == 2 and dur.shape[0] == 1


@pytest.mark.requires_models
def test_empty_text():
    try:
        tts().synthesize("   ")
    except SystemExit as e:
        assert "empty text" in str(e.code)
    else:
        raise AssertionError("empty text must fail")


@pytest.mark.requires_models
def test_unknown_speaker():
    try:
        tts().synthesize("Привет.", speaker="nope")
    except ValueError as e:
        assert "nope" in str(e) and "xenia" in str(e)
    else:
        raise AssertionError("unknown speaker must fail with voice list")


@pytest.mark.requires_models
def test_bad_sample_rate():
    try:
        tts().synthesize("Привет.", sample_rate=16000)
    except SystemExit as e:
        assert "48000" in str(e.code)
    else:
        raise AssertionError("bad sample rate must fail")


def test_missing_models_dir():
    try:
        SileroTTS(model_dir=os.path.join(ROOT, "no-such-dir"),
                  version="v5_ru")
    except SystemExit as e:
        assert "acoustic_v5_ru.onnx" in str(e.code)
    else:
        raise AssertionError("missing models dir must fail")


def test_factory_unknown_version():
    from silero_onnx.strategies import create_strategy
    try:
        create_strategy(version="v9")
    except ValueError as e:
        assert "v9" in str(e) and "v5_ru" in str(e)
    else:
        raise AssertionError("unknown version must fail with the list")


def test_factory_v56_missing_models(tmp_path):
    # v5_6 assets absent from the given directory: the factory must fail
    # loudly, naming the expected files (no silent fallback to v5_ru).
    # tmp_path is empty by construction, so this test never depends on
    # what the local build output happens to contain.
    from silero_onnx.strategies import create_strategy
    try:
        create_strategy(version="v5_6_ru", model_dir=str(tmp_path))
    except SystemExit as e:
        assert "acoustic_v5_6_ru.onnx" in str(e.code), str(e.code)
    else:
        raise AssertionError("missing v5_6 assets must fail loudly")


def test_zzz_no_torch_at_end():
    assert "torch" not in sys.modules


if __name__ == "__main__":
    ordered = [("test_no_torch_at_import", test_no_torch_at_import)]
    ordered += [(k, v) for k, v in sorted(globals().items())
                if k.startswith("test_") and callable(v)
                and k not in ("test_no_torch_at_import", "test_zzz_no_torch_at_end")]
    ordered.append(("test_zzz_no_torch_at_end", test_zzz_no_torch_at_end))
    failed = 0
    for name, fn in ordered:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(ordered) - failed}/{len(ordered)} passed")
    raise SystemExit(1 if failed else 0)
