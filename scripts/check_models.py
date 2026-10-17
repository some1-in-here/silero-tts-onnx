"""Integrity check for a built model directory (default: <repo>/models/onnx/).

Verifies, without torch:
  1. every .onnx model every supported version needs exists and loads in
     onnxruntime (CPU) with the expected I/O names. Coverage is enforced,
     not assumed: EXPECTED must contain every MODEL_SETS artifact (both
     the v5_ru set and the v5_6_ru set), or this script fails up front,
  2. multi-shape ORT robustness: seeded numpy inputs of varying T/B/W
     run clean (finite outputs, no hangs) — crash-guard, no torch ref,
  3. a short end-to-end synthesis works through SileroTTS for BOTH
     versions (v5_ru and the default v5_6_ru).

Deployment utility: run once after scripts/export_all.py (or whenever
models/onnx/ is refreshed). NOT part of the runtime; never called by
tests.

Usage:
  python scripts/check_models.py [--models models/onnx]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# I/O spec per artifact (names AND order are the runtime contract).
EXPECTED = {
    # -- v5_ru set (5 files) --
    "acoustic_v5_ru.onnx": (["sequence", "speaker_ids", "durs_rate",
                             "pitch_coefs"], ["mel", "dur"]),
    "vocoder_v5_ru.onnx": (["mel"], ["mag", "x", "y"]),
    "istft_48k.onnx": (["mag", "x", "y"], ["audio"]),
    "silero_stress.onnx": (["input_ids", "homo_start_ids",
                            "homo_end_ids"], ["logits"]),
    "ngram_core.onnx": (["ind", "offsets"],
                        ["stress_logits", "yo_logits"]),
    # -- v5_6_ru set (4 files; stress/ngram above are shared) --
    "acoustic_v5_6_ru.onnx": (["sequence", "speaker_ids", "durs_rate",
                               "gt_durs", "gt_pitch", "type_ids",
                               "focus_mask"], ["mel", "dur"]),
    "dur_predictor_v5_6_ru.onnx": (["text", "speaker_ids", "mask",
                                    "type_ids", "focus_mask"], ["dur"]),
    "pitch_predictor_v5_6_ru.onnx": (["text", "speaker_ids", "mask",
                                      "type_ids", "focus_mask"],
                                     ["pitch"]),
    "vocoder_v5_6_ru.onnx": (["mel"], ["audio"]),
}

ROBUST_SEED = 42


def _robustness(sessions, failures):
    """Seeded multi-shape ORT runs (numpy only): finite outputs, no hangs."""
    import numpy as np

    rng = np.random.RandomState(ROBUST_SEED)
    total, bad = 0, 0

    def run(sess, feeds, tag):
        nonlocal total, bad
        total += 1
        try:
            outs = sess.run(None, feeds)
        except Exception as e:
            failures.append(tag + " ORT run failed: " + str(e)[:200])
            bad += 1
            return
        shapes = ",".join(str(tuple(o.shape)) for o in outs)
        if not all(np.all(np.isfinite(o)) for o in outs):
            failures.append(tag + " non-finite output " + shapes)
            bad += 1
            return
        print("robust " + tag + " " + shapes, flush=True)

    # -- v5_ru sessions -------------------------------------------------
    for T in (12, 45, 130):
        run(sessions["acoustic_v5_ru.onnx"],
            {"sequence": rng.randint(0, 47, (1, T)).astype(np.int64),
             "speaker_ids": np.array([3], dtype=np.int64),
             "durs_rate": np.ones((1, T), dtype=np.float32),
             "pitch_coefs": np.ones((1, T), dtype=np.float32)},
            "acoustic T=%d" % T)
    for T in (24, 200, 511):
        run(sessions["vocoder_v5_ru.onnx"],
            {"mel": (rng.randn(1, 192, T) * 0.5).astype(np.float32)},
            "vocoder T=%d" % T)
    for T in (16, 90, 300):
        run(sessions["istft_48k.onnx"],
            {"mag": (np.abs(rng.randn(1, 1201, T)) * 0.5).astype(np.float32),
             "x": (rng.randn(1, 1201, T) * 0.5).astype(np.float32),
             "y": (rng.randn(1, 1201, T) * 0.5).astype(np.float32)},
            "istft T=%d" % T)

    # -- shared frontend sessions ---------------------------------------
    with open(os.path.join(ROOT, "silero_onnx", "data", "tokenizer.json"),
              encoding="utf-8") as f:
        n_vocab = len(json.load(f)["vocab"])
    for B, T in ((1, 20), (3, 48), (2, 33)):
        st = [int(rng.randint(1, T - 1)) for _ in range(B)]
        en = [int(min(T - 1, s + rng.randint(1, 5))) for s in st]
        run(sessions["silero_stress.onnx"],
            {"input_ids": rng.randint(0, n_vocab, (B, T)).astype(np.int64),
             "homo_start_ids": np.array(st, dtype=np.int64),
             "homo_end_ids": np.array(en, dtype=np.int64)},
            "stress B=%d/T=%d" % (B, T))
    from silero_onnx.ngram_accentor import OnnxNgram
    probe = OnnxNgram.__new__(OnnxNgram)
    with open(os.path.join(ROOT, "silero_onnx", "data", "ngrams.json"),
              encoding="utf-8") as f:
        probe.ngram_dict = json.load(f)
    from silero_onnx.ngram_accentor import featurize
    pool = ["замок", "был", "открыт", "меня", "зовут", "лева", "королев",
            "и", "уже", "готов", "привет", "мир"]
    for W in (1, 5, 12):
        words = [pool[i % len(pool)] for i in rng.randint(0, len(pool), W)]
        ind, offs = featurize(words, probe.ngram_dict)
        run(sessions["ngram_core.onnx"], {"ind": ind, "offsets": offs},
            "ngram W=%d" % W)

    # -- v5_6_ru sessions -------------------------------------------------
    # Feeds mirror the engine operating point: type_ids < 12 for the dur
    # predictor, base ids (ti % 6) for pitch, gt_durs linear, gt_pitch raw,
    # focus_mask in {0, 3} (see scripts/strategies/v5_6/export.py).
    for T in (12, 45, 130):
        seq = rng.randint(0, 47, (1, T)).astype(np.int64)
        spk = np.array([3], dtype=np.int64)
        ti = rng.randint(0, 12, (1, T)).astype(np.int64)
        fm = rng.choice([0, 3], size=(1, T)).astype(np.int64)
        mask = seq == 0
        run(sessions["dur_predictor_v5_6_ru.onnx"],
            {"text": seq, "speaker_ids": spk, "mask": mask,
             "type_ids": ti, "focus_mask": fm},
            "dur56 T=%d" % T)
        run(sessions["pitch_predictor_v5_6_ru.onnx"],
            {"text": seq, "speaker_ids": spk, "mask": mask,
             "type_ids": ti % 6, "focus_mask": fm},
            "pitch56 T=%d" % T)
        run(sessions["acoustic_v5_6_ru.onnx"],
            {"sequence": seq, "speaker_ids": spk,
             "durs_rate": np.ones((1, T), dtype=np.float32),
             "gt_durs": rng.uniform(0.5, 4.0, (1, T)).astype(np.float32),
             # rank 3, like the pitch predictor's own output (1, 1, T)
             "gt_pitch": rng.rand(1, 1, T).astype(np.float32),
             "type_ids": ti, "focus_mask": fm},
            "acoustic56 T=%d" % T)
    for T in (24, 200, 400):
        run(sessions["vocoder_v5_6_ru.onnx"],
            {"mel": (rng.randn(1, 192, T) * 0.5).astype(np.float32)},
            "vocoder56 T=%d" % T)

    print("robustness: %d runs (seed %d), %d bad"
          % (total, ROBUST_SEED, bad), flush=True)


def main():
    ap = argparse.ArgumentParser(description="check built ONNX model dir")
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "onnx"))
    a = ap.parse_args()

    failures = []

    # Coverage guard: MODEL_SETS is the single source of truth for what a
    # build must contain. A newly added set/artifact must never slip past
    # this check silently.
    from silero_onnx.versions import MODEL_SETS  # torch-free
    want = {f for files in MODEL_SETS.values() for f in files.values()}
    uncovered = sorted(want - set(EXPECTED))
    if uncovered:
        print("check_models FAILED: no I/O spec for " + ", ".join(uncovered)
              + " (add them to EXPECTED: every MODEL_SETS artifact must be "
                "validated here)", flush=True)
        raise SystemExit(1)
    print("MODEL_SETS coverage: all %d artifacts have I/O specs" % len(want),
          flush=True)

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR only (see silero_onnx/engine.py)

    sessions = {}
    for name, spec in EXPECTED.items():
        want_in, want_out = spec
        base = os.path.join(a.models, name)
        if not os.path.isfile(base):
            failures.append("missing " + base)
            continue
        try:
            sess = ort.InferenceSession(
                base, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e:
            failures.append("cannot load " + name + ": " + str(e))
            continue
        got_in = [i.name for i in sess.get_inputs()]
        got_out = [o.name for o in sess.get_outputs()]
        if got_in != want_in or got_out != want_out:
            failures.append(
                name + " I/O mismatch: in=" + str(got_in) +
                " out=" + str(got_out))
            continue
        sessions[name] = sess
        print("OK " + name, flush=True)

    if not failures:
        _robustness(sessions, failures)

    if not failures:
        from silero_onnx import SileroTTS
        import numpy as np
        for version, text in (("v5_ru", "Привет."),
                              ("v5_6_ru", "Привет! Меня зовут **Ксения**.")):
            tts = SileroTTS(model_dir=a.models, version=version)
            audio, _dur = tts.synthesize(text, speaker="xenia")
            assert audio.dtype == np.float32 and np.all(np.isfinite(audio))
            assert len(audio) > 1000, len(audio)
            print("OK smoke synthesis %s (%d samples)"
                  % (version, len(audio)), flush=True)

    if failures:
        print("check_models FAILED:", flush=True)
        for f in failures:
            print("  - " + f, flush=True)
        raise SystemExit(1)
    print("check_models: all %d models + smoke tests (v5_ru + v5_6_ru) "
          "passed" % len(EXPECTED), flush=True)


if __name__ == "__main__":
    main()
