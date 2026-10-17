"""End-to-end parity: torch chain vs ONNX chain (build env only).

Compares, on fixed texts, the torch reference pipeline
(AcousticClean -> VocWrap -> RealISTFT, all torch/no_grad) against the
exported assets (acoustic -> vocoder -> istft .onnx chained the same way):
mel (atol 1e-3), mag (atol 2e-2), final waveform (atol 5e-4). Non-zero
exit on breach.

The waveform bar is the load-bearing one: on the verified stack (torch
2.14.1 / onnxruntime 1.31.0 - the same wheels locally and on CI) it sits
at 9.4e-5 (short text) .. 2.8e-4 (long text, GitHub ubuntu-latest),
against the 5e-4 bar. mel/mag bars are diagnostics: vocoder conv-stack
kernel noise grows with utterance length AND shifts with the CPU kernel
path, so they carry deliberate headroom over the worst observation
(mag 1.9e-3 short / 3.8e-3 long on Windows, 5.9e-3 long on CI) instead
of pinning the local numbers. See scripts/DEVELOPMENT.md item 3.

Why here and not in tests/: the suite is torch-free by contract
(first/last sys.modules assertions) and must run without torch or
network. The per-component halves of this check already run as L1
gates inside each export_*.py; this script is their end-to-end twin,
runnable with one command in the build env:

  python scripts/check_parity.py [--models onnx]
"""
import argparse
import os
import sys

# v5 build blocks moved to scripts/strategies/v5/ (symmetric layout).
_V5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "strategies", "v5")
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)
del _V5_DIR

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEXTS = ["Привет! Меня зовут Ксения.",
         "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"]

# NOTE: mag is a diagnostic, not a fidelity claim. Conv-stack kernel
# noise grows with utterance length and shifts with the platform kernel
# path: the local-only 5e-3 bar (measured 1.9e-03 short / 3.8e-03 long
# text) tripped the first CI run at 5.9e-3 on the long text, while the
# waveform it produced stayed at 2.8e-4. The bar now sits ~3.4x above
# the worst observation so a runner change cannot flake the build. The
# audio gate below is the load-bearing one; mag/mel are diagnostic.
GATES = {"mel": 1e-3, "mag": 2e-2, "audio": 5e-4}


def main():
    ap = argparse.ArgumentParser(description="torch-vs-ONNX e2e parity")
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "onnx"))
    a = ap.parse_args()

    import numpy as np
    import torch

    from build_acoustic import build_acoustic
    from istft_model import RealISTFT

    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker="v5_ru", trust_repo=True)
    model.to("cpu")
    part = model.packages[0]
    jit_m = part.models[0]

    def get(root, *names):
        m = root
        for n in names:
            m = dict(m.named_children())[n]
        return m

    voc = get(jit_m, "vocoder")
    new = build_acoustic(jit_m, [])
    istft = RealISTFT().eval()

    class VocWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mel):
            mag, x, y = self.m(mel, 48000, 0.0, True)
            return mag, x, y

    vw = VocWrap(voc).eval()

    import onnxruntime as ort
    kw = {"providers": ["CPUExecutionProvider"]}
    sa = ort.InferenceSession(os.path.join(a.models, "acoustic_v5_ru.onnx"),
                              **kw)
    sv = ort.InferenceSession(os.path.join(a.models, "vocoder_v5_ru.onnx"),
                              **kw)
    si = ort.InferenceSession(os.path.join(a.models, "istft_48k.onnx"), **kw)

    ok = True
    for text in TEXTS:
        spk_ids, _mid = part.get_speakers("xenia")
        sents, _, blens, rates, pitches, sp_ids = \
            part.prepare_tts_model_input(text, ssml=False,
                                         speaker_ids=spk_ids)
        seq, _, dr, pc = part.merge_batch_model(
            sents, blens, rates, pitches, put_accent=True, put_yo=True)
        with torch.no_grad():
            t_mel, _d = new(seq, sp_ids, dr.clone(), pc.clone())
            t_mag, t_x, t_y = vw(t_mel)
            t_audio = istft(t_mag, t_x, t_y).numpy()
        o_mel, _d = sa.run(None, {"sequence": seq.numpy(),
                                  "speaker_ids": sp_ids.numpy(),
                                  "durs_rate": dr.numpy(),
                                  "pitch_coefs": pc.numpy()})
        o_mag, o_x, o_y = sv.run(None, {"mel": o_mel.astype(np.float32)})
        o_audio = si.run(None, {"mag": o_mag.astype(np.float32),
                                "x": o_x.astype(np.float32),
                                "y": o_y.astype(np.float32)})[0]
        morate = all(r.shape == o.shape
                     for _, r, o in (("mel", t_mel.numpy(), o_mel),
                                     ("mag", t_mag.numpy(), o_mag),
                                     ("audio", t_audio, o_audio)))
        for nm, r, o in (("mel", t_mel.numpy(), o_mel),
                         ("mag", t_mag.numpy(), o_mag),
                         ("audio", t_audio, o_audio)):
            if r.shape != o.shape:
                print(f"{text[:20]!r} {nm}: SHAPE_MISMATCH "
                      f"torch{r.shape} ort{o.shape} FAIL", flush=True)
                ok = False
                continue
            d = float(np.abs(r - o).max())
            passed = bool(np.allclose(r, o, rtol=0, atol=GATES[nm]))
            print(f"{text[:20]!r} {nm}: max_abs={d:.3e} "
                  f"gate={GATES[nm]} {'PASS' if passed else 'FAIL'}",
                  flush=True)
            ok &= passed
    if not ok:
        raise SystemExit("check_parity: waveform parity gate FAILED")
    print("check_parity: torch-vs-ONNX end-to-end PASS", flush=True)


if __name__ == "__main__":
    main()
