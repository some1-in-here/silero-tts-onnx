"""Export the homograph solver -> silero_stress.onnx (single file).

Source: silero_stress accentor homosolver BERT-tiny (`load_accentor()`).
NOTE: the upstream object is a TorchScript module (RecursiveScriptModule),
which torch.export/dynamo refuses ("Exporting a ScriptModule is not
supported"), so this exporter uses the CLASSIC torchscript path
(torch.onnx.export with dynamo=False) — the natural fit for scripted
graphs. Dynamic B+T axes via dynamic_axes.

Inputs : input_ids (B,T) int64, homo_start_ids (B,) int64,
         homo_end_ids (B,) int64.  B (1..32) and T (1..512) are dynamic.
Output : logits (B,1) fp32 variant scores (sigmoid + round in runtime).

The reference batch (7 real homograph windows) is generated, not
versioned: <out-dir>/fixtures/stress_ref.pt via scripts/make_fixtures.py
(run from the repo root).

Gates: L1 onnx.checker + ORT vs torch max_abs <= 1e-4 + variant agreement
(sigmoid-round must pick the same homograph variant) + a B=3/T=13 probe;
then L2 stochastic (5 seeded random batches, max_abs only, atol=1e-3).

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export_stress.py [--out-dir onnx]

Result: <out-dir>/silero_stress.onnx.
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
                     collapse_call, finish, fixtures_dir, seed_all,
                     stochastic_probe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination dir for silero_stress.onnx")
    a = ap.parse_args()

    import torch

    from silero_stress import load_accentor

    log = Logger()
    begin(log, "export_stress.py", "silero_stress.onnx")
    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "silero_stress.onnx")

    acc = load_accentor()
    m = acc.homosolver.model.eval()
    fx_file = os.path.join(fixtures_dir(a.out_dir), "stress_ref.pt")
    if not os.path.isfile(fx_file):
        raise SystemExit(
            "export_stress: missing fixture " + fx_file + " — run "
            "python scripts/make_fixtures.py (or export_all.py) first")
    ref = torch.load(fx_file, weights_only=True)
    ids, st, en = ref["ids"], ref["starts"], ref["ends"]
    with torch.no_grad():
        torch_ref = m(ids, st, en)
    log.log(f"ref logits {tuple(torch_ref.shape)}")

    # NOTE: classic path (scripted input) + opset 17: the proven combination
    # (vocoder exports the same way). The opset-18 requirement is dynamo-only.
    collapse_call(torch.onnx.export,
        m, (ids, st, en), onnx_path,
        input_names=["input_ids", "homo_start_ids", "homo_end_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "B", 1: "T"},
                      "homo_start_ids": {0: "B"},
                      "homo_end_ids": {0: "B"},
                      "logits": {0: "B"}},
        opset_version=17, dynamo=False)
    log.log("classic export=OK")

    sess, ok = check_and_gate(
        onnx_path,
        {"input_ids": ids.numpy(), "homo_start_ids": st.numpy(),
         "homo_end_ids": en.numpy()},
        [torch_ref.numpy()], ["logits"], log.log, atol=1e-4)

    import numpy as np
    o = sess.run(None, {"input_ids": ids.numpy(),
                        "homo_start_ids": st.numpy(),
                        "homo_end_ids": en.numpy()})[0]
    pred_torch = torch.sigmoid(torch_ref).round().int().tolist()
    pred_ort = (1 / (1 + np.exp(-o))).round().astype(int).tolist()
    agree = pred_torch == pred_ort
    log.log(f"variants torch={pred_torch} ort={pred_ort} agree={agree}")
    ok &= agree

    ids2, st2, en2 = ids[:3, :13], st[:3], en[:3]
    with torch.no_grad():
        r2 = m(ids2, st2, en2).numpy()
    try:
        o2 = sess.run(None, {"input_ids": ids2.numpy(),
                             "homo_start_ids": st2.numpy(),
                             "homo_end_ids": en2.numpy()})[0]
        if r2.shape != o2.shape:
            log.log("L1 B3/T13 SHAPE_MISMATCH")
            ok = False
        else:
            log.log(f"L1 B3/T13 {o2.shape}: max_abs={np.abs(r2 - o2).max():.3e}")
    except Exception as e:
        log.log(f"L1 B3 FAIL {type(e).__name__}: {str(e)[:200]}")
        ok = False

    # Level-2 (stochastic robustness): seeded random id batches with valid
    # homo spans, varying B/T. Relaxed (1e-3); random spans are content-free,
    # so only max_abs is gated here (no variant agreement — that is L1).
    seed_all()
    vmax = int(ids.max()) + 1  # in-embedding-range bound, from the L1 batch
    l2_cases = []
    with torch.no_grad():
        for Bb, Tt in ((1, 16), (2, 32), (4, 48), (1, 24), (3, 64)):
            cids = torch.randint(0, vmax, (Bb, Tt))
            cst, cen = [], []
            for _ in range(Bb):
                s = int(np.random.randint(1, Tt - 1))
                e = int(min(Tt - 1, s + np.random.randint(1, 5)))
                cst.append(s)
                cen.append(e)
            cst = torch.LongTensor(cst)
            cen = torch.LongTensor(cen)
            cr = m(cids, cst, cen).numpy()
            l2_cases.append(([cr], {"input_ids": cids.numpy(),
                                    "homo_start_ids": cst.numpy(),
                                    "homo_end_ids": cen.numpy()}))
    ok &= stochastic_probe(sess, l2_cases, ["logits"], log.log, atol=1e-3)

    finish(log, os.path.join(a.out_dir, "stress_export_result.txt"),
           [onnx_path], "export_stress", ok)


if __name__ == "__main__":
    main()
