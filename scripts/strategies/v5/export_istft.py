"""Export the inverse STFT -> istft_48k.onnx (single file).

Source: RealISTFT (istft_model.py), a real-valued port of the v5
ISTFT (n_fft=2400, hop=600, win=2400, padding="same"). The inverse DFT is
spelled out with precomputed cos/sin bases in float64 (float32
accumulation over 2400 terms was measured at ~1e-4 error; float64 keeps
it at ~1e-7), so no ComplexFloat ever crosses the ONNX boundary.

Inputs : mag, x, y (B,1201,T) fp32, T dynamic; spec = mag*(x + i*y).
Output : audio (B,N) fp32 waveform (48 kHz path, no PQMF).

Gates:
  - RealISTFT vs torch.fft ISTFT on a real T=57 reference + a synthetic
    T=9 case: max_abs <= 1e-4 (else a non-zero SystemExit aborts before export),
  - onnx.checker + L1 ORT vs RealISTFT (atol=1e-4), plus a second-length
    probe; then L2 stochastic (5 seeded random spectra, atol=1e-3).

The T-reference comes from the generated fixture
(<out-dir>/fixtures/spec_ref.pt, via scripts/make_fixtures.py) — run from
the repo root so relative paths resolve.

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export_istft.py [--out-dir onnx]

Result: <out-dir>/istft_48k.onnx (weights inlined, single file).
"""
import argparse
import os
import sys

# Path bootstrap for `python scripts/strategies/v5/<this>.py`
# (v5 blocks live next to this file; shared _common in scripts/).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(os.path.dirname(_HERE)), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _HERE, _p
from _common import (DEFAULT_OUT_DIR, Logger, begin, check_and_gate,
                     collapse_call, finish, fixtures_dir,
                     inline_external_data, seed_all, stochastic_probe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination dir for istft_48k.onnx")
    a = ap.parse_args()

    import torch

    from istft_model import RealISTFT

    log = Logger()
    begin(log, "export_istft.py", "istft_48k.onnx (single file)")
    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "istft_48k.onnx")

    new = RealISTFT().eval()
    fx_file = os.path.join(fixtures_dir(a.out_dir), "spec_ref.pt")
    if not os.path.isfile(fx_file):
        raise SystemExit(
            "export_istft: missing fixture " + fx_file + " — run "
            "python scripts/make_fixtures.py (or export_all.py) first")
    ref = torch.load(fx_file, weights_only=True)
    cases = {
        ("real%d" % ref["mag"].shape[2]): (ref["mag"], ref["x"], ref["y"]),
        "synth9": (torch.rand(1, 1201, 9), torch.randn(1, 1201, 9),
                   torch.randn(1, 1201, 9)),
    }
    # Fidelity gate against torch.fft: need the original ISTFT params.
    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker="v5_ru", trust_repo=True)
    wv = model.packages[0].wrapped_jit_v
    assert (wv.istft.n_fft, wv.istft.hop_length, wv.istft.win_length) == \
        (2400, 600, 2400), "upstream ISTFT params changed: port must be re-based"
    allok = True
    with torch.no_grad():
        for cname, (mag, x, y) in cases.items():
            spec = mag * (x + 1j * y)
            r1 = wv.istft(spec)
            r2 = new(mag, x, y)
            assert tuple(r1.shape) == tuple(r2.shape), (r1.shape, r2.shape)
            d = float((r1 - r2).abs().max())
            ok = bool(torch.allclose(r1, r2, atol=1e-4))
            allok &= ok
            log.log(f"{cname}{tuple(r2.shape)}: max_abs={d:.3e} ok1e-4={ok}")
    if not allok:
        log.write(os.path.join(a.out_dir, "istft_export_result.txt"))
        raise SystemExit("export_istft: RealISTFT fidelity gate FAILED")

    real_key = sorted(k for k in cases if k.startswith("real"))[0]
    mag, x, y = cases[real_key]
    T = torch.export.Dim("T", min=1, max=4096)
    # NOTE: opset 18 (torch 2.14 has no exporter impls below 18; see
    # export_acoustic.py). ORT>=1.17 loads it fine.
    collapse_call(torch.onnx.export,
        new, (mag, x, y), onnx_path,
        input_names=["mag", "x", "y"], output_names=["audio"],
        dynamic_shapes={"mag": {2: T}, "x": {2: T}, "y": {2: T}},
        opset_version=18, dynamo=True)
    log.log("dynamo export=OK")

    inline_external_data(onnx_path, log.log)
    with torch.no_grad():
        refr = new(mag, x, y).numpy()
    sess, ok = check_and_gate(
        onnx_path,
        {"mag": mag.numpy(), "x": x.numpy(), "y": y.numpy()},
        [refr], ["audio"], log.log, atol=1e-4)

    m2, x2, y2 = cases["synth9"]
    with torch.no_grad():
        refr2 = new(m2, x2, y2).numpy()
    try:
        o2 = sess.run(None, {"mag": m2.numpy(), "x": x2.numpy(),
                             "y": y2.numpy()})[0]
        if refr2.shape != o2.shape:
            log.log(f"L1 T9 SHAPE_MISMATCH torch={refr2.shape} ort={o2.shape}")
            ok = False
        else:
            import numpy as np
            log.log(f"L1 T9 torch={refr2.shape} ort={o2.shape} "
                    f"max_abs={np.abs(refr2 - o2).max():.3e}")
    except Exception as e:
        log.log(f"L1 T9 FAIL {type(e).__name__}: {str(e)[:200]}")
        ok = False

    # Level-2 (stochastic robustness): seeded random spectra, varying T.
    # Relaxed (1e-3): the float64 core keeps real-data error at ~1e-7, but
    # noise inputs are held to a looser bar on purpose (breakage guard).
    seed_all()
    l2_cases = []
    with torch.no_grad():
        for Tcase in (9, 32, 100, 257, 17):
            cmg = torch.abs(torch.randn(1, 1201, Tcase))
            cx = torch.randn(1, 1201, Tcase)
            cy = torch.randn(1, 1201, Tcase)
            cr = new(cmg, cx, cy).numpy()
            l2_cases.append(([cr], {"mag": cmg.numpy(), "x": cx.numpy(),
                                    "y": cy.numpy()}))
    ok &= stochastic_probe(sess, l2_cases, ["audio"], log.log, atol=1e-3)

    finish(log, os.path.join(a.out_dir, "istft_export_result.txt"),
           [onnx_path], "export_istft", ok)


if __name__ == "__main__":
    main()
