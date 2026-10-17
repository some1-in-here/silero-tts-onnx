"""Extended robustness suite for silero_onnx (torch-free).

Covers: text edge cases, thread safety (6 threads, shared instance),
memory stability (50 sequential runs, RSS via ctypes), audio integrity
(NaN/Inf/range/RMS), fault injection (bad speaker/paths/corrupt data),
RTF benchmark. Uses ONLY the package + stdlib/numpy/soundfile —
never torch and never build-time helpers (asserted explicitly).

Run:  py -3.12 -m pytest tests/test_extended_robustness.py -v
  or: py -3.12 tests/test_extended_robustness.py
"""
import concurrent.futures
import gc
import os
import sys
import tempfile
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

assert "torch" not in sys.modules, "test process imported torch before package!"

from silero_onnx import env  # noqa: E402

from silero_onnx import SileroTTS  # noqa: E402
from silero_onnx import frontend as F  # noqa: E402

def _default_models():
    """Model dir: SILERO_ONNX_MODELS override, else the local build
    output (<repo>/models/onnx). Model tests are marked and skipped
    with a reason when the set is absent (see tests/conftest.py)."""
    envdir = os.environ.get(env.MODELS_DIR_VAR)
    if envdir:
        return envdir
    return os.path.normpath(os.path.join(ROOT, "models", "onnx"))


MODELS = _default_models()
SHORT = "Привет."
MEDIUM = "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"

_tts = None


def tts():
    global _tts
    if _tts is None:
        _tts = SileroTTS(model_dir=MODELS, version="v5_ru")
    return _tts


def _valid_audio(a, min_len=1):
    assert isinstance(a, np.ndarray) and a.dtype == np.float32
    assert a.ndim == 1 and len(a) >= min_len
    assert np.all(np.isfinite(a)), "NaN/Inf in audio"


def test_aaa_no_torch_at_import():
    assert "torch" not in sys.modules


@pytest.mark.requires_models
def test_emoji_latin_specials():
    audio, _dur = tts().synthesize("Привет! Hello 😀 #тег @user 100% !!!")
    _valid_audio(audio, 48000)


@pytest.mark.requires_models
def test_negative_and_fractional_numbers():
    audio, _dur = tts().synthesize("Температура -5 градусов, было 3,5 кг.")
    _valid_audio(audio, 48000)
    # minus must expand (not vanish into a hyphen), decimals digit-wise
    norm = F.normalize("Температура -5 градусов, было 3,5 кг.")
    assert "минус пять" in norm and "три целых пять" in norm


@pytest.mark.requires_models
def test_punctuation_glue():
    # no spaces after punctuation: must not crash (matches original:
    # no space insertion, comma/excl are regular symbols)
    audio, _dur = tts().synthesize("Привет,мир!Как дела?Хорошо.")
    _valid_audio(audio, 48000)


@pytest.mark.requires_models
def test_long_multiparagraph():
    para = ("В недрах тундры выдры в гетрах тырят в вёдра ядра кедров. "
            "Теплый хлеб пахнет домом, а капли дождя стучат по стеклу.\n")
    long_text = para * 3
    audio_long, _d = tts().synthesize(long_text)
    audio_short, _d = tts().synthesize(SHORT)
    _valid_audio(audio_long, 48000)
    assert len(audio_long) > 3 * len(audio_short)


@pytest.mark.requires_models
def test_degenerate_strings():
    for bad in ("", "   ", "\n\t  \n"):
        try:
            tts().synthesize(bad)
        except SystemExit as e:
            assert "empty text" in str(e.code)
        else:
            raise AssertionError(f"degenerate {bad!r} must fail cleanly")
    # punctuation-only: dots are valid symbols -> short but valid audio
    audio, _dur = tts().synthesize("...")
    _valid_audio(audio, 1)


@pytest.mark.requires_models
def test_thread_safety_6x():
    t = tts()
    t.synthesize(SHORT)  # warmup (lazy ORT sessions + tokenizer cache first)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(t.synthesize, SHORT) for _ in range(6)]
        outs = [f.result(timeout=300) for f in futs]
    for audio, dur in outs:
        _valid_audio(audio, 1)  # SHORT is ~0.7 s; length gate below is separate
        assert dur.ndim == 2 and dur.shape[0] == 1
    lens = sorted(len(a) for a, _d in outs)
    assert lens[-1] - lens[0] <= 1200, f"length spread too big: {lens}"
    # pairwise numerics only within equal-length runs (a ±1-frame boundary
    # flip shifts tails and would fake a mismatch; lengths already gated)
    base = outs[0][0]
    n = min(len(a) for a, _d in outs)
    same_len = [a for a, _d in outs if len(a) == len(base)]
    assert len(same_len) >= 4, f"too few equal-length runs: {[len(a) for a, _d in outs]}"
    for audio in same_len[1:]:
        d = np.abs(base[:n] - audio[:n]).max()
        assert d <= 1e-2, f"thread-run divergence {d:.3e}"


def _rss_mb():
    """Process RSS in MB. Windows via psapi (explicit signatures);
    falls back to /proc on POSIX; None when unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            # NOTE: PageFaultCount is DWORD (4 bytes), the rest SIZE_T.
            # Getting this wrong shifts every field (observed: bogus 0.1 MB).
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                          ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.WorkingSetSize / 1e6
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1e3
    except Exception:
        pass
    return None


@pytest.mark.requires_models
def test_memory_50x():
    t = tts()
    for _ in range(3):
        t.synthesize(SHORT)  # warmup: stabilize ORT arenas + caches
    gc.collect()
    rss0 = _rss_mb()
    for i in range(50):
        audio, dur = t.synthesize(SHORT)
        assert dur.shape[0] == 1 and np.all(np.isfinite(audio))
        s = float(np.abs(audio).sum())  # touch the buffer (no dead code)
        assert s > 0
        if i % 10 == 9:
            gc.collect()
    gc.collect()
    rss1 = _rss_mb()
    if rss0 is None or rss1 is None:
        print("SKIP rss check (process counters unavailable on this platform)")
        return
    print(f"RSS before={rss0:.0f}MB after={rss1:.0f}MB delta={rss1 - rss0:.0f}MB")
    assert rss1 - rss0 < 200, f"possible leak: +{rss1 - rss0:.0f}MB over 50 runs"


@pytest.mark.requires_models
def test_audio_integrity():
    audio, _dur = tts().synthesize(MEDIUM)
    assert np.all(np.isfinite(audio)), "NaN/Inf found"
    peak = float(np.abs(audio).max())
    assert peak <= 1.0, f"clipping: peak {peak}"
    rms = float((audio.astype(np.float64) ** 2).mean() ** 0.5)
    print(f"peak={peak:.4f} rms={rms:.4f}")
    assert 0.005 < rms < 0.6, f"bad energy level: rms {rms}"


@pytest.mark.requires_models
def test_fault_unknown_speaker_case():
    try:
        tts().synthesize("Привет.", speaker="XENIA")
    except ValueError as e:
        assert "XENIA" in str(e) and "xenia" in str(e)
    else:
        raise AssertionError("wrong-case speaker must fail with voice list")


def test_fault_empty_model_files():
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("acoustic_v5_ru.onnx", "vocoder_v5_ru.onnx", "istft_48k.onnx"):
            open(os.path.join(tmp, f), "wb").close()  # present but empty/broken
        try:
            SileroTTS(model_dir=tmp, version="v5_ru")
        except SystemExit as e:
            assert "sidecar" in str(e.code) or "cannot load" in str(e.code), str(e.code)
        else:
            raise AssertionError("broken models dir must fail")


def test_fault_corrupt_data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            from silero_onnx.frontend import TorchFreeFrontend
            TorchFreeFrontend(models_dir=MODELS, data_dir=tmp)
        except SystemExit as e:
            assert "frontend data" in str(e.code), str(e.code)
        else:
            raise AssertionError("corrupt data dir must fail cleanly")


@pytest.mark.requires_models
def test_rtf_benchmark():
    t = tts()
    t.synthesize(MEDIUM)  # warmup
    times, spoken = [], 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        audio, _dur = t.synthesize(MEDIUM)
        dt = time.perf_counter() - t0
        times.append(dt)
        spoken = len(audio) / 48000
    rtf = spoken / min(times)
    print(f"audio={spoken:.2f}s best_wall={min(times):.2f}s RTF={rtf:.1f}x")
    assert rtf > 1.0, f"slower than realtime: {rtf}"


def test_zzz_no_torch_at_end():
    assert "torch" not in sys.modules


if __name__ == "__main__":
    ordered = [("test_aaa_no_torch_at_import", test_aaa_no_torch_at_import)]
    ordered += [(k, v) for k, v in sorted(globals().items())
                if k.startswith("test_") and callable(v)
                and k not in ("test_aaa_no_torch_at_import",
                              "test_zzz_no_torch_at_end")]
    ordered.append(("test_zzz_no_torch_at_end", test_zzz_no_torch_at_end))
    failed = 0
    for name, fn in ordered:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(ordered) - failed}/{len(ordered)} passed")
    raise SystemExit(1 if failed else 0)
