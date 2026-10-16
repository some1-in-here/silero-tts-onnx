"""Export the acoustic model -> acoustic_v5_ru.onnx (single file).

Source: snakers4/silero-models (ru/v5_ru) Tacotron side, transplanted into
the clean port (build_acoustic.build_acoustic, 224 tensors) and exported
with the dynamo exporter (classic torchscript export can no longer lower
this graph on current torch: layer_norm/add collapse to rank-unknown
onnx:: nodes and the following transposes fail).

Inputs : sequence (B,T) int64, speaker_ids (B,) int64,
         durs_rate (B,T) fp32, pitch_coefs (B,T) fp32.  T is dynamic
         (batch is fixed to 1: the punkt/len_reg wiring is single-
         utterance by construction, see acoustic.py).
Outputs: mel (B,192,F) fp32, dur (B,T) int64 (dur is float32 pre-round
         inside, int64 after torch.round — dtype preserved by dynamo).

Gates (fail the script when violated):
  - transplanted tensor count == 224 (inside build_acoustic),
  - L1 strict: torch dur vs original pipeline dur max_abs <= 1e-4, then
    ORT vs torch (mel, dur) on the fixed T=32 reference, atol=1e-3,
  - same ORT session re-run on a longer (T=73) input,
  - L2 stochastic: 5 seeded random token feeds (T=8..160), ORT vs torch
    through the same session, atol=3e-3 (robustness, not fidelity).

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export_acoustic.py [--out-dir onnx]

Result: <out-dir>/acoustic_v5_ru.onnx (weights inlined, single file).
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
from _common import (DEFAULT_OUT_DIR, ROOT, Logger, begin, check_and_gate,
                     collapse_call, finish, inline_external_data, seed_all,
                     stochastic_probe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination dir for acoustic_v5_ru.onnx")
    a = ap.parse_args()

    import numpy as np
    import torch

    from build_acoustic import build_acoustic

    log = Logger()
    begin(log, "export_acoustic.py", "acoustic_v5_ru.onnx (single file)")
    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "acoustic_v5_ru.onnx")

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

    new = build_acoustic(jit_m, [])  # asserts 224 transplanted tensors

    # Reference inputs from the REAL frontend (not synthetic): T=32 text.
    text = "Привет! Меня зовут Ксения."
    spk_ids, mid = part.get_speakers("xenia")
    sents, _, blens, rates, pitches, sp_ids = part.prepare_tts_model_input(
        text, ssml=False, speaker_ids=spk_ids)
    seq, _, dr, pc = part.merge_batch_model(sents, blens, rates, pitches,
                                            put_accent=True, put_yo=True)
    with torch.no_grad():
        out, out_add = part.models[mid](sequence=seq, speaker_ids=sp_ids, sr=48000,
                                        symb_durs={}, durs_rate=dr.clone(),
                                        pitch_coefs=pc.clone(), device="cpu")
        mel_new, dur_new = new(seq, sp_ids, dr.clone(), pc.clone())
    dd = float((dur_new - out_add).abs().max())
    log.log(f"torch dur vs orig pipeline: max_abs={dd:.3e}")
    if dd > 1e-4:
        raise SystemExit(f"export_acoustic: torch-level dur mismatch {dd:.3e}")

    # Dynamo export (onnxscript-based): the classic torchscript exporter
    # can no longer lower this graph on current torch (rank-unknown
    # onnx:: nodes break the transposes). The module is dynamo-ready by
    # construction: branch-free clamps, no .item(), batch fixed to 1.
    T = torch.export.Dim("T", min=1, max=2048)
    # NOTE: opset 18 is requested, not 17: torch 2.14 only ships exporter
    # implementations for 18, and the 18->17 version-converter pass fails
    # on this graph (kept at 18 anyway). ORT>=1.17 loads opset 18 fine.
    collapse_call(torch.onnx.export,
        new, (seq, sp_ids, dr.clone(), pc.clone()), onnx_path,
        input_names=["sequence", "speaker_ids", "durs_rate", "pitch_coefs"],
        output_names=["mel", "dur"],
        # NOTE: only INPUTS may be listed here (outputs inherit dynamism);
        # mel frames (F) derive from durations and come out dynamic as well.
        # Every input must appear as a key (static ones map to {}).
        dynamic_shapes={"sequence": {1: T}, "speaker_ids": {},
                        "durs_rate": {1: T}, "pitch_coefs": {1: T}},
        opset_version=18, dynamo=True)
    log.log("dynamo export=OK")

    inline_external_data(onnx_path, log.log)
    sess, ok = check_and_gate(
        onnx_path,
        {"sequence": seq.numpy(), "speaker_ids": sp_ids.numpy(),
         "durs_rate": dr.numpy(), "pitch_coefs": pc.numpy()},
        [mel_new.numpy(), dur_new.numpy()], ["mel", "dur"], log.log, atol=1e-3)

    # Dynamic-T probe through the SAME session (different utterance length).
    text2 = "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"
    sents2, _, blens2, rates2, pitches2, sp_ids2 = part.prepare_tts_model_input(
        text2, ssml=False, speaker_ids=spk_ids)
    seq2, _, dr2, pc2 = part.merge_batch_model(sents2, blens2, rates2, pitches2,
                                               put_accent=True, put_yo=True)
    with torch.no_grad():
        mel2, dur2 = new(seq2, sp_ids2, dr2.clone(), pc2.clone())
    try:
        outs2 = sess.run(None, {"sequence": seq2.numpy(),
                                "speaker_ids": sp_ids2.numpy(),
                                "durs_rate": dr2.numpy(),
                                "pitch_coefs": pc2.numpy()})
        for nm, r, o in (("mel", mel2.numpy(), outs2[0]),
                         ("dur", dur2.numpy(), outs2[1])):
            if r.shape != o.shape:
                log.log(f"L1 T73 {nm}: SHAPE_MISMATCH torch={r.shape} ort={o.shape}")
                ok = False
            else:
                log.log(f"L1 T73 {nm}{o.shape}: "
                        f"max_abs={np.abs(r - o).max():.3e}")
    except Exception as e:
        log.log(f"L1 T73 FAIL {type(e).__name__}: {str(e)[:300]}")
        ok = False

    # Level-2 (stochastic robustness): seeded random token feeds with
    # varying T through the SAME session. Threshold is relaxed (3e-3):
    # random ids exercise odd mask/tail paths; this guards breakage,
    # fidelity is level-1's job.
    seed_all()
    l2_cases = []
    with torch.no_grad():
        for Tcase in (8, 24, 48, 96, 160):
            cids = torch.randint(0, 47, (1, Tcase))
            cids[0, 0] = 2
            csp = torch.randint(0, 5, (1,))
            crt = torch.ones(1, Tcase)
            cpc = torch.ones(1, Tcase)
            cm, cd = new(cids, csp, crt.clone(), cpc.clone())
            l2_cases.append(([cm.numpy(), cd.numpy()],
                             {"sequence": cids.numpy(),
                              "speaker_ids": csp.numpy(),
                              "durs_rate": crt.numpy(),
                              "pitch_coefs": cpc.numpy()}))
    ok &= stochastic_probe(sess, l2_cases, ["mel", "dur"], log.log,
                           atol=3e-3)

    finish(log, os.path.join(a.out_dir, "acoustic_export_result.txt"),
           [onnx_path], "export_acoustic", ok)


if __name__ == "__main__":
    main()
