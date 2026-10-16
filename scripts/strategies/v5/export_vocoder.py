"""Export the vocoder -> vocoder_v5_ru.onnx (single file, no sidecar).

Source: snakers4/silero-models (ru/v5_ru) vocoder submodule, wrapped so the
fixed call arguments (sample_rate=48000, ...) become graph constants.

Input : mel (B,192,T) fp32, T dynamic (frames).
Outputs: mag (B,1201,F), x (B,1201,F), y (B,1201,F) — split spectral parts;
         the waveform is reconstructed downstream by istft_48k.onnx, so no
         complex tensor ever crosses an ONNX boundary.

Gates: onnx.checker + L1 ORT vs torch on a REAL mel (deterministic
in-distribution reference, see below), allclose atol=1e-3; then L2
stochastic: 5 seeded randn mels (T=16..320), atol=3e-3.

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export_vocoder.py [--out-dir onnx]

Result: <out-dir>/vocoder_v5_ru.onnx.
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
                     collapse_call, finish, seed_all, stochastic_probe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination dir for vocoder_v5_ru.onnx")
    a = ap.parse_args()

    import torch

    class VocWrap(torch.nn.Module):
        """Freeze the vocoder call signature to a single mel input."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mel):
            mag, x, y = self.m(mel, 48000, 0.0, True)
            return mag, x, y

    log = Logger()
    begin(log, "export_vocoder.py", "vocoder_v5_ru.onnx")
    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "vocoder_v5_ru.onnx")

    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker="v5_ru", trust_repo=True)
    jit_m = model.packages[0].models[0]

    def get(root, *names):
        m = root
        for n in names:
            m = dict(m.named_children())[n]
        return m

    voc = get(jit_m, "vocoder")
    mel_dim = get(jit_m, "tacotron", "lin").weight.shape[0]
    log.log(f"mel_dim={mel_dim}")
    assert mel_dim == 192, mel_dim

    # Deterministic in-distribution reference: a REAL mel produced by the
    # clean acoustic on the fixed T32 text — never synthetic randn (its
    # ORT-vs-torch diff varies run to run: 2.0e-04 once, 3.5e-04 next,
    # turning the gate into a lottery).
    from build_acoustic import build_acoustic
    part = model.packages[0]
    new = build_acoustic(jit_m, [])
    text = "Привет! Меня зовут Ксения."
    spk_ids, _mid = part.get_speakers("xenia")
    sents, _, blens, rates, pitches, sp_ids = part.prepare_tts_model_input(
        text, ssml=False, speaker_ids=spk_ids)
    seq, _, dr, pc = part.merge_batch_model(sents, blens, rates, pitches,
                                            put_accent=True, put_yo=True)
    with torch.no_grad():
        mel, _dur = new(seq, sp_ids, dr.clone(), pc.clone())
        ref = voc(mel, 48000, 0.0, True)
    log.log(f"ref mel {tuple(mel.shape)} from clean acoustic (deterministic)")

    traced = torch.jit.trace(VocWrap(voc), (mel,), strict=False)
    collapse_call(torch.onnx.export,
        traced, (mel,), onnx_path,
        input_names=["mel"], output_names=["mag", "x", "y"],
        dynamic_axes={"mel": {2: "T"}, "mag": {2: "T"},
                      "x": {2: "T"}, "y": {2: "T"}},
        opset_version=17, dynamo=False)

    # NOTE: gate is 1e-3, not 1e-4. Measured on torch 2.14 / ORT 1.31:
    # real mel 7.7e-04, seeded randn 2.3e-04..5.6e-04 on mag — the deep
    # conv stack accumulates kernel-level numeric noise between torch and
    # ORT BLAS, systematically above 1e-4 on ANY input. 1e-3 keeps the
    # gate meaningful (two orders below audibility-relevant drift);
    # end-to-end waveform fidelity is guarded by the runtime suite.
    sess, ok = check_and_gate(onnx_path, {"mel": mel.numpy()},
                              [t.numpy() for t in ref],
                              ["mag", "x", "y"], log.log, atol=1e-3)

    # Level-2 (stochastic robustness): seeded randn mels, varying frames.
    # Relaxed (3e-3): noise accumulates conv-stack error vs real spectra.
    seed_all()
    l2_cases = []
    with torch.no_grad():
        for Tcase in (16, 57, 200, 320, 48):
            cm = (torch.randn(1, mel_dim, Tcase) * 0.5).float()
            cr = voc(cm, 48000, 0.0, True)
            l2_cases.append(([t.numpy() for t in cr], {"mel": cm.numpy()}))
    ok &= stochastic_probe(sess, l2_cases, ["mag", "x", "y"], log.log,
                           atol=3e-3)
    finish(log, os.path.join(a.out_dir, "vocoder_export_result.txt"),
           [onnx_path], "export_vocoder", ok)


if __name__ == "__main__":
    main()
