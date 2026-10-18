"""Model-free contracts for the additive multilingual path (no torch)."""
import json
from pathlib import Path

import numpy as np
import pytest

from silero_onnx.multilingual import MultilingualRuntime


def data():
    symbols = '_~| !"\',-.:;?abcdefghijklmnopqrstuvwxyz'
    return dict(symbols=symbols, symbol_to_id={c: i for i, c in enumerate(symbols)},
                speaker_to_id={"en_0": 0}, symb_ascii_dict={}, modern=False,
                phons=False, spectral=False, sos="|", eos="~", sample_rate=48000)


def test_plain_frontend_preserves_original_symbols():
    d = data()
    feed = MultilingualRuntime.prepare_data(d, "Hello, WORLD!", "en_0", "en")
    decoded = "".join(d["symbols"][i] for i in feed["sequence"][0])
    assert decoded == "|hello, world!~"
    assert feed["sequence"].dtype == np.int64
    assert feed["speaker_ids"].tolist() == [0]
    assert feed["durs_rate"].dtype == np.float32
    np.testing.assert_array_equal(feed["durs_rate"], feed["pitch_coefs"])


@pytest.mark.parametrize("text", ["", "123", "!?", "\u202a\ufffc"])
def test_unsupported_text_raises_explicitly(text):
    with pytest.raises(ValueError, match="no supported text"):
        MultilingualRuntime.prepare_data(data(), text, "en_0")


@pytest.mark.parametrize("voice", ["random", "unknown"])
def test_unknown_or_random_voice_rejected(voice):
    with pytest.raises(ValueError, match="Unknown fixed voice"):
        MultilingualRuntime.prepare_data(data(), "hello", voice)


def test_input_limit():
    MultilingualRuntime.prepare_data(data(), "a" * 1000, "en_0")
    with pytest.raises(ValueError, match="chunk it first"):
        MultilingualRuntime.prepare_data(data(), "a" * 1001, "en_0")


def test_phonetic_input_not_silently_assumed():
    d = data()
    d["phons"] = True
    with pytest.raises(NotImplementedError):
        MultilingualRuntime.prepare_data(d, "hello", "en_0")


def test_modern_frontend_boundaries_and_newline():
    symbols = "_~| !',-.:;?^абвгдежзийклмнопрстуфхцчшщыэюя"
    d = data()
    d.update(symbols=symbols, symbol_to_id={c: i for i, c in enumerate(symbols)},
             speaker_to_id={"ru_0": 0}, alphabet=list(symbols), ext_alph={},
             modern=True)
    feed = MultilingualRuntime.prepare_data(d, "Привет\nмир!", "ru_0", "ru")
    decoded = "".join(symbols[i] for i in feed["sequence"][0])
    assert decoded == "|привет. мир!~"


@pytest.mark.parametrize("low", [False, True])
def test_options_are_scoped_to_istft(tmp_path, monkeypatch, low):
    d = data()
    d["spectral"] = True
    (tmp_path / "frontend.json").write_text(json.dumps(d), encoding="utf-8")
    import onnxruntime as ort
    seen = []

    def session(path, opts, providers):
        seen.append((Path(path).name, opts.graph_optimization_level,
                     opts.enable_cpu_mem_arena, opts.enable_mem_pattern, providers))
        return object()

    monkeypatch.setattr(ort, "InferenceSession", session)
    MultilingualRuntime(tmp_path, lower_memory=low)
    assert [v[0] for v in seen] == ["acoustic.onnx", "vocoder.onnx", "istft.onnx"]
    assert all(v[1] == ort.GraphOptimizationLevel.ORT_ENABLE_ALL for v in seen[:2])
    assert seen[2][1] == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert all(v[2:4] == (not low, not low) for v in seen)
    assert all(v[4] == ["CPUExecutionProvider"] for v in seen)


def test_reject_nonvalidated_sample_rate(tmp_path):
    d = data()
    d["sample_rate"] = 24000
    (tmp_path / "frontend.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="48000"):
        MultilingualRuntime(tmp_path)


def test_synthesis_chain_and_finite_gate():
    class Session:
        def __init__(self, values):
            self.values = values
        def run(self, outputs, feeds):
            return self.values
    tts = MultilingualRuntime.__new__(MultilingualRuntime)
    tts.data = data()
    tts.acoustic = Session([np.zeros((1, 1, 1)), np.ones((1, 3))])
    tts.vocoder = Session([np.array([[[0.1, 0.2]]], dtype=np.float32)])
    tts.istft = None
    audio, durations = tts.synthesize("hello", "en_0")
    assert audio.shape == (2,)
    assert durations.shape == (1, 3)
    tts.vocoder.values = [np.array([np.nan])]
    with pytest.raises(ValueError, match="non-finite"):
        tts.synthesize("hello", "en_0")


def test_export_rejects_disabled_validation():
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-O", str(root / "scripts/export_multilingual.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "do not use Python -O" in result.stderr


def test_package_allowlist_hashes_and_no_overwrite(tmp_path):
    import hashlib
    import subprocess
    import sys
    import zipfile
    root = Path(__file__).resolve().parent.parent
    d = data()
    inventory = json.loads((root / "scripts/multilingual-checkpoints.json").read_text(encoding="utf-8"))
    d.update(id="v3_en", source_sha256=inventory["v3_en"]["sha256"])
    d["graphs"] = {}
    for name in ("acoustic.onnx", "vocoder.onnx"):
        content = b"mock graph for packaging test, not an inference fixture"
        (tmp_path / name).write_bytes(content)
        d["graphs"][name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    (tmp_path / "frontend.json").write_text(json.dumps(d), encoding="utf-8")
    (tmp_path / "private.log").write_text("must not be bundled", encoding="utf-8")
    output = tmp_path / "package.zip"
    command = [sys.executable, str(root / "scripts/package_multilingual.py"),
               "--model-dir", str(tmp_path), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    with zipfile.ZipFile(output) as z:
        assert set(z.namelist()) == {"acoustic.onnx", "vocoder.onnx", "frontend.json",
                                    "MODEL-LICENSE.txt", "ATTRIBUTION.md", "SOURCE.json"}
        assert z.testzip() is None
    assert subprocess.run(command, capture_output=True).returncode != 0
    (tmp_path / "acoustic.onnx").write_bytes(b"corrupt")
    output.unlink()
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "Graph hash/size mismatch" in failed.stderr
