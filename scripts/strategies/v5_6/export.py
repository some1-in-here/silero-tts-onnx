"""Export v5_6_ru -> acoustic/dur/pitch/vocoder onnx (single files).

Isolated per-version orchestrator: everything v5_6 lives in scripts/strategies/v5_6/
and never touches the v5 exporters. Output paths come EXCLUSIVELY from
silero_onnx.versions.MODEL_SETS["v5_6_ru"] (no hardcoded filenames).

Route (measured, not assumed): direct lowering of the v5_6 TorchScript
graphs is impossible on torch 2.14 — three independent blockers, each
reproduced: TS2EPConverter dies on prim::If with list mutations (freeze
makes it worse: 0 -> 192 Ifs); direct dynamo refuses ScriptModules;
classic export dies on prim::is_nested (predictors) and complex-tensor
ISTFT (vocoder). A fourth: the fold-based ISTFT tail does not
dynamo-export INSIDE the fused graph (data-dependent guard), though it
exports fine standalone (like v5's istft). So this follows the
v5-proven clean-room discipline: transplant weights into eager ports
(series.py/forward.py/vocoder.py via build.py, reusing the
proven v5 blocks) and export those with dynamo (opset 18) — with the
pipeline SPLIT at the mel boundary (a real intermediate of the torch
graph): acoustic (text->mel,dur) + vocoder (mel->audio), exactly the v5
session topology extended. Fidelity is proven by gates, not counts.

MEASURED deviations from the v5 track:
- acoustic_v5_6_ru.onnx maps (sequence, speaker_ids, durs_rate,
  gt_durs, gt_pitch, type_ids, focus_mask) -> (mel, dur).
  pitch_coefs has NO input: with gt fed, the scripted graph ignores it
  (gt_pitch bypasses update_pitch_coef), so prosody must be folded
  into gt_pitch engine-side; gt_durs arrives LINEAR (exp(pred)-1);
  durs_rate still applies (two-pass break
  forcing works through it).
- Uniform always-gt orchestration: the engine ALWAYS feeds gt_* from
  the standalone predictor sessions (full-sequence call). gt_durs is
  LINEAR domain (engine feeds exp(pred)-1, as torch's own windowed
  path does — feeding log output shrinks durations ~e-fold); gt_pitch
  is RAW predictor output. Export and gates validate exactly this
  operating point.
- Pitch predictor type space is 6, dur is 12: shifted ids (>= 6) OOB
  the pitch embedding — the engine must feed it base ids (ti % 6),
  exactly what torch does via base_type_ids. Gated, not assumed.
- Predictor embedding rows: type 12/6, emphasis 2, focus 4, emotion
  dead (0/15, unused in forward math, not transplanted).
- emphasis input is never fed by the package (bakes absent everywhere);
  dur emphasis_emb flag is False, pitch True-but-unfed.
- PQMF resamplers exist but only trigger at 24k/8k: dead at sr=48000,
  not ported.
- symbols 47=47, same 5 speakers/dims -> stress/ngram shared with v5.
- The punkt-zero loop is provably dead (loop bound is 0 for every
  input) and omitted, as in v5.

Steps (each gated, fail loud): load_source -> capture_feeds (spy proxy
records EXACT torch feeds incl. emphasis texts) -> transplant (L0:
predictors/mel/dur strict torch-vs-torch; joint audio INFO) ->
dynamo export x4 -> L1 ORT gates (acoustic mel/dur, predictors direct,
vocoder on scripted-tacotron mel) -> L2 stochastic -> e2e info.
Result: 4 single-file models + v5_6_export_result.txt.

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5_6/export.py [--out-dir onnx]
"""
import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.dirname(THIS_DIR)
SCRIPTS_DIR = os.path.dirname(STRATEGIES_DIR)
V5_DIR = os.path.join(STRATEGIES_DIR, "v5")
ROOT = os.path.dirname(SCRIPTS_DIR)
# THIS_DIR: v5_6 ports; V5_DIR: shared v5 blocks (same arch, reused
# without duplication); SCRIPTS_DIR: _common; ROOT: runtime package.
for _p in (THIS_DIR, V5_DIR, SCRIPTS_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (DEFAULT_OUT_DIR, Logger, begin, check_and_gate,
                     collapse_call, finish, inline_external_data,
                     package_data_path, seed_all, stochastic_probe,
                     write_speakers_json)
from silero_onnx.versions import MODEL_SETS  # torch-free: safe in build env

from build import build_acoustic56, build_main, transplant_predictors, transplant_tacotron, transplant_vocoder

MODEL_TAG = "v5_6_ru"
HUB_SPEAKER = "v5_6_ru"
FILES = MODEL_SETS[MODEL_TAG]

# Bars: L0/L1 are same-math (tight); L2 matches v5 track conventions.
# (Joint clean-vs-scripted audio is INFO-only: the float64-real ISTFT
# port vs torch complex ISTFT diverge at ~1e-4..1e-3 rel on loud
# passages; same-math ORT parity is the vocoder gate.)
# gt convention (mirrors torch exactly): gt_durs is LINEAR domain
# (engine feeds exp(pred)-1, as torch's own windowed path does);
# gt_pitch is RAW predictor output (no exp anywhere on the pitch path).
GATE_L0_PRED = 1e-6
GATE_L0_MEL = 1e-5
GATE_MAIN_DUR = 1e-6
GATE_PRED = 1e-4
GATE_VOC = 1e-3

CASES = [
    ("plain", "Привет! Меня зовут Ксения."),
    ("plain2", "Меня зовут Лева Королев. Я уже готов открыть все ваши замки!"),
    ("exclam", "Меня зовут Ксения!"),
    ("emph2", "Привет! Меня зовут **Ксения**."),
    ("emph1", "Меня зовут **Ксения**."),
    ("focus1", "Меня зовут *Ксения*."),
]


def load_source(log):
    """torch.hub v5_6 (no force_reload: hub files may be file-locked)."""
    import torch

    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker=HUB_SPEAKER, trust_repo=True)
    model.to("cpu")
    part = model.packages[0]
    assert len(part.speakers) >= 1, part.speakers
    log.log("source OK: %s, speakers=%s" % (HUB_SPEAKER, part.speakers))
    # Speaker map for the torch-free runtime (names AND ids from the
    # package's own get_speakers(); write-or-validate against the
    # committed table — a changed table fails loudly here).
    entries = [(name, *part.get_speakers(name)) for name in part.speakers]
    entries = [(name, ids[0], mid) for name, ids, mid in entries]
    write_speakers_json(entries, package_data_path("speakers.json"),
                        log.log)
    return part, part.models[0]


def _get(root, *names):
    m = root
    for n in names:
        m = dict(m.named_children())[n]
    return m


class _Spy:
    """Transparent proxy over the scripted main: records call kwargs for
    the parity gates, then delegates. Attribute access passes through,
    so apply_tts runs its exact torch path."""

    def __init__(self, jm):
        self._jm = jm
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_jm"], name)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.__dict__["_jm"](*args, **kwargs)


def _linear_dur(pred):
    """Predictor log-dur -> gt_durs linear convention (torch.exp - 1).

    Mirrors torch's own gt construction (ts lines 822/859): the main
    graph consumes gt_durs WITHOUT exp, so the engine must exponentiate.
    Feeding raw log output would shrink all durations ~e-fold (caught by
    the e2e length check, not by shape-identical L1 feeds)."""
    import torch

    return torch.exp(pred) - 1


def capture_feeds(part, jm, log):
    """Run the REAL torch frontend per case text; record exact main feeds.

    feeds are pre-call CLONES (the torch call mutates gt in place); torch
    outputs are recomputed on those feeds. Missing gt is completed with
    full-sequence torch predictors (dur: recorded ids; pitch: base ids,
    mirroring torch's base_type_ids) — the engine operating point."""
    import torch

    out = []
    mid = part.get_speakers("xenia")[1]
    real = part.models[mid]
    spy = _Spy(jm)
    part.models[mid] = spy
    try:
        with torch.no_grad():
            for tag, text in CASES:
                spy.calls.clear()
                part.apply_tts(text=text, speaker="xenia", sample_rate=48000,
                               lang="ru")
                assert len(spy.calls) == 1, (tag, len(spy.calls))
                _a, kw = spy.calls[0]
                feeds = {k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in kw.items()}
                if feeds.get("gt_durs") is None or \
                        feeds.get("gt_pitch") is None:
                    seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
                    ti, fm = feeds["type_ids"], feeds["focus_mask"]
                    mask = torch.zeros(1, seq.shape[1], dtype=torch.bool)
                    if feeds.get("gt_durs") is None:
                        feeds["gt_durs"] = _linear_dur(_get(
                            jm, "dur_predictor")(
                                seq, sp_ids, mask, 1.0, None, ti, fm))
                    if feeds.get("gt_pitch") is None:
                        feeds["gt_pitch"] = _get(
                            jm, "pitch_predictor")(
                                seq, sp_ids, mask,
                                torch.remainder(ti, 6), fm)
                    log.log("L1 %s: torch left gt to the engine; "
                            "gate uses full-seq torch gt" % tag)
                t_audio, t_dur = jm(**feeds)
                out.append((tag, text, feeds, t_audio.numpy(),
                            t_dur.numpy()))
                log.log("L1 %s: T=%d audio%s dur%s gt=%s" %
                        (tag, feeds["sequence"].shape[1],
                         tuple(t_audio.shape), tuple(t_dur.shape),
                         "recorded" if kw.get("gt_durs") is not None
                         else "full-seq"))
    finally:
        part.models[mid] = real
    return out


def transplant_all(jm, log):
    """Clean-room modules + transplant proof (counts add to measured)."""
    ld = []
    dur, pitch = transplant_predictors(jm, ld)
    log.log("transplant predictors: %d tensors" % len(ld))
    assert len(ld) == 118, len(ld)
    lt, lv = [], []
    taco = transplant_tacotron(jm, lt)
    voc = transplant_vocoder(jm, lv)
    log.log("transplant tacotron: %d, vocoder: %d tensors"
            % (len(lt), len(lv)))
    assert len(lt) + len(lv) == 193, (len(lt), len(lv))
    from forward import Acoustic56, Main56
    acoustic = Acoustic56(taco).eval()
    main = Main56(acoustic, voc).eval()
    return dur, pitch, acoustic, voc, main


def l0_predictors(jm, dur, pitch, log, captured):
    """L0: clean predictors vs scripted originals (same math => ~1e-7)."""
    import numpy as np
    import torch

    ok = True
    with torch.no_grad():
        for tag, _text, feeds, _ta, _td in captured:
            seq, sp_ids, ti, fm = (feeds["sequence"], feeds["speaker_ids"],
                                   feeds["type_ids"], feeds["focus_mask"])
            mask = (seq == 0)
            mask_b = torch.zeros(1, seq.shape[1], dtype=torch.bool)
            for key, clean in (("dur_predictor", dur),
                               ("pitch_predictor", pitch)):
                mod = _get(jm, key)
                pti = ti if key == "dur_predictor" else torch.remainder(ti, 6)
                if key == "dur_predictor":
                    ref = mod(seq, sp_ids, mask_b, 1.0, None, pti,
                              fm).numpy()
                else:
                    ref = mod(seq, sp_ids, mask_b, pti, fm).numpy()
                got = clean(seq, sp_ids, mask, pti, fm).numpy()
                d = float(np.abs(ref - got).max()) \
                    if ref.shape == got.shape else float("inf")
                passed = d <= GATE_L0_PRED
                log.log("L0 %s/%s: max_abs=%.3e ok%g=%s"
                        % (key, tag, d, GATE_L0_PRED, passed))
                ok &= passed
    return ok


def l0_main(jm, main, log, captured):
    """L0 joint (chain proof): clean acoustic+vocoder vs scripted main.

    INFO-only for audio (the float64-real ISTFT port vs torch complex
    ISTFT diverge at ~1e-4..1e-3 relative on loud passages — same-math
    ORT parity is the L1 vocoder gate below); dur is gated exact.
    Localizes instantly: l0_acoustic below proves the tacotron half."""
    import numpy as np
    import torch

    ok = True
    with torch.no_grad():
        for tag, _text, feeds, t_audio, t_dur in captured:
            seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
            o_audio, o_dur = main(seq, sp_ids, feeds["durs_rate"],
                                  feeds["gt_durs"], feeds["gt_pitch"],
                                  feeds["type_ids"], feeds["focus_mask"])
            o_audio, o_dur = o_audio.numpy(), o_dur.numpy()
            da = float(np.abs(t_audio - o_audio).max()) \
                if t_audio.shape == o_audio.shape else float("inf")
            dd = float(np.abs(t_dur - o_dur).max()) \
                if t_dur.shape == o_dur.shape else float("inf")
            rel = da / max(1e-9, float(np.abs(t_audio).max()))
            pd = dd <= GATE_MAIN_DUR
            log.log("L0 main/%s: audio max_abs=%.3e rel=%.3e (info); "
                    "dur max_abs=%.3e ok%g=%s"
                    % (tag, da, rel, dd, GATE_MAIN_DUR, pd))
            ok &= pd
    return ok


def l0_acoustic(jm, acoustic, log, captured):
    """L0: clean acoustic (mel, dur) vs scripted tacotron-direct mel +
    scripted main dur on matched feeds. Strict: same math throughout."""
    import numpy as np
    import torch

    ok = True
    with torch.no_grad():
        for tag, _text, feeds, _ta, t_dur in captured:
            seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
            t_mel = _get(jm, "tacotron")(seq, sp_ids, (seq == 0),
                                         torch.from_numpy(t_dur),
                                         feeds["gt_pitch"], None).numpy()
            o_mel, o_dur = acoustic(seq, sp_ids, feeds["durs_rate"],
                                    feeds["gt_durs"], feeds["gt_pitch"],
                                    feeds["type_ids"], feeds["focus_mask"])
            o_mel, o_dur = o_mel.numpy(), o_dur.numpy()
            dm = float(np.abs(t_mel - o_mel).max()) \
                if t_mel.shape == o_mel.shape else float("inf")
            dd = float(np.abs(t_dur - o_dur).max()) \
                if t_dur.shape == o_dur.shape else float("inf")
            pm = dm <= GATE_L0_MEL
            pd = dd <= GATE_MAIN_DUR
            log.log("L0 acoustic/%s: mel%s max_abs=%.3e ok%g=%s; "
                    "dur max_abs=%.3e ok%g=%s"
                    % (tag, o_mel.shape, dm, GATE_L0_MEL, pm, dd,
                       GATE_MAIN_DUR, pd))
            ok &= pm and pd
    return ok


def dynamo_export(module, sample, path, in_names, out_names, dyn, log, what):
    """Eager clean module -> single-file ONNX (tensor-only, dynamo)."""
    import torch

    collapse_call(torch.onnx.export, module, sample, path,
                  input_names=in_names, output_names=out_names,
                  dynamic_shapes=dyn, opset_version=18, dynamo=True)
    log.log("dynamo export %s=OK" % what)
    inline_external_data(path, log.log)
    import onnx
    onnx.checker.check_model(path)
    log.log("checker %s=OK" % what)


def export_all_onnx(dur, pitch, acoustic, voc, out_dir, log, captured):
    """Export the 4 sessions (predictors, acoustic mel-model, vocoder)."""
    import torch

    _tag, _text, feeds, _ta, _td = captured[0]
    seq, sp_ids, ti, fm = (feeds["sequence"], feeds["speaker_ids"],
                           feeds["type_ids"], feeds["focus_mask"])
    mask = (seq == 0)
    TD = torch.export.Dim("T", min=1, max=2048)
    paths = {}

    paths["dur_predictor"] = os.path.join(out_dir, FILES["dur_predictor"])
    dynamo_export(dur, (seq, sp_ids, mask, ti, fm),
                  paths["dur_predictor"],
                  ["text", "speaker_ids", "mask", "type_ids", "focus_mask"],
                  ["dur"],
                  {"text": {1: TD}, "speaker_ids": {}, "mask": {1: TD},
                   "type_ids": {1: TD}, "focus_mask": {1: TD}},
                  log, "dur_predictor")

    paths["pitch_predictor"] = os.path.join(out_dir, FILES["pitch_predictor"])
    dynamo_export(pitch, (seq, sp_ids, mask, ti, fm),
                  paths["pitch_predictor"],
                  ["text", "speaker_ids", "mask", "type_ids", "focus_mask"],
                  ["pitch"],
                  {"text": {1: TD}, "speaker_ids": {}, "mask": {1: TD},
                   "type_ids": {1: TD}, "focus_mask": {1: TD}},
                  log, "pitch_predictor")

    paths["acoustic"] = os.path.join(out_dir, FILES["acoustic"])
    dynamo_export(acoustic,
                  (seq, sp_ids, feeds["durs_rate"], feeds["gt_durs"],
                   feeds["gt_pitch"], ti, fm),
                  paths["acoustic"],
                  ["sequence", "speaker_ids", "durs_rate", "gt_durs",
                   "gt_pitch", "type_ids", "focus_mask"],
                  ["mel", "dur"],
                  {"sequence": {1: TD}, "speaker_ids": {},
                   "durs_rate": {1: TD}, "gt_durs": {1: TD},
                   "gt_pitch": {2: TD}, "type_ids": {1: TD},
                   "focus_mask": {1: TD}},
                  log, "acoustic")

    # Standalone vocoder (mel->audio): in-distribution sample mel from
    # the clean acoustic itself.
    paths["vocoder"] = os.path.join(out_dir, FILES["vocoder"])
    with torch.no_grad():
        mref = acoustic(seq, sp_ids, feeds["durs_rate"], feeds["gt_durs"],
                        feeds["gt_pitch"], ti, fm)[0].detach()
    log.log("vocoder sample mel %s (clean acoustic, in-distribution)"
            % (tuple(mref.shape),))
    FD = torch.export.Dim("F", min=8, max=4096)
    dynamo_export(voc, (mref,), paths["vocoder"], ["mel"], ["audio"],
                  {"mel": {2: FD}}, log, "vocoder")
    return paths


def parity_acoustic(jm, out_dir, log, captured):
    """L1: ORT acoustic (mel, dur) vs scripted tacotron-direct mel +
    scripted main dur on matched gt feeds (emphasis texts incl.)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sa = ort.InferenceSession(os.path.join(out_dir, FILES["acoustic"]),
                              providers=["CPUExecutionProvider"])
    ok = True
    for tag, _text, feeds, _ta, t_dur in captured:
        seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
        with torch.no_grad():
            t_mel = _get(jm, "tacotron")(
                seq, sp_ids, (seq == 0), torch.from_numpy(t_dur),
                feeds["gt_pitch"], None).numpy()
        ort_feeds = {k: feeds[k].numpy()
                     for k in ("sequence", "speaker_ids", "durs_rate",
                               "gt_durs", "gt_pitch", "type_ids",
                               "focus_mask")}
        o_mel, o_dur = sa.run(None, ort_feeds)
        dm = float(np.abs(t_mel - o_mel).max()) \
            if t_mel.shape == o_mel.shape else float("inf")
        dd = float(np.abs(t_dur - o_dur).max()) \
            if t_dur.shape == o_dur.shape else float("inf")
        pm = dm <= GATE_PRED
        pd = dd <= GATE_MAIN_DUR
        log.log("L1 acoustic/%s: mel%s max_abs=%.3e ok%g=%s; dur%s "
                "max_abs=%.3e ok%g=%s"
                % (tag, o_mel.shape, dm, GATE_PRED, pm,
                   o_dur.shape, dd, GATE_MAIN_DUR, pd))
        ok &= pm and pd
    return ok, sa


def parity_predictors(jm, out_dir, log, captured):
    """L1 direct: scripted-vs-ORT predictors on recorded inputs."""
    import numpy as np
    import onnxruntime as ort
    import torch

    ok = True
    for key in ("dur_predictor", "pitch_predictor"):
        sess = ort.InferenceSession(
            os.path.join(out_dir, FILES[key]),
            providers=["CPUExecutionProvider"])
        mod = _get(jm, key)
        with torch.no_grad():
            for tag, _text, feeds, _ta, _td in captured:
                seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
                ti, fm = feeds["type_ids"], feeds["focus_mask"]
                mask = (seq == 0)
                mask_b = torch.zeros(1, seq.shape[1], dtype=torch.bool)
                pti = ti if key == "dur_predictor" else torch.remainder(ti, 6)
                if key == "dur_predictor":
                    ref = mod(seq, sp_ids, mask_b, 1.0, None, pti,
                              fm).numpy()
                else:
                    ref = mod(seq, sp_ids, mask_b, pti, fm).numpy()
                got = sess.run(None, {"text": seq.numpy(),
                                      "speaker_ids": sp_ids.numpy(),
                                      "mask": mask.numpy(),
                                      "type_ids": pti.numpy(),
                                      "focus_mask": fm.numpy()})[0]
                d = float(np.abs(ref - got).max()) \
                    if ref.shape == got.shape else float("inf")
                passed = d <= GATE_PRED
                log.log("L1 %s/%s: %s max_abs=%.3e ok%g=%s"
                        % (key, tag, got.shape, d, GATE_PRED, passed))
                ok &= passed
    return ok


def parity_vocoder(jm, out_dir, log, captured):
    """L1 vocoder: scripted-vs-ORT on REAL tacotron mel (direct scripted
    The mel comes from a direct scripted tacotron call fed with the
    torch main's own (dur_hat7, gt_pitch) — in-distribution by
    construction, no hooks needed (unsupported on ScriptModules)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(os.path.join(out_dir, FILES["vocoder"]),
                                providers=["CPUExecutionProvider"])
    voc = _get(jm, "vocoder")
    ok = True
    with torch.no_grad():
        for tag, _text, feeds, _ta, t_dur in captured[:3]:
            seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
            t_mel = _get(jm, "tacotron")(seq, sp_ids, (seq == 0),
                                         torch.from_numpy(t_dur),
                                         feeds["gt_pitch"], None)
            ref = voc(t_mel, 48000, 0.0, True).numpy()
            got = sess.run(None, {"mel": t_mel.numpy()})[0]
            d = float(np.abs(ref - got).max()) \
                if ref.shape == got.shape else float("inf")
            passed = d <= GATE_VOC
            log.log("L1 vocoder/%s: mel%s max_abs=%.3e ok%g=%s"
                    % (tag, tuple(t_mel.shape), d, GATE_VOC, passed))
            ok &= passed
    seed_all()
    l2 = []
    # In-distribution mels (NOT randn): randn spectra drive exp(mag)
    # into a regime where torch-vs-ORT differ at 1e-2 (measured) — an
    # OOD artifact, not export breakage. Real mels from random-token
    # torch runs vary length/distribution while staying in-domain.
    with torch.no_grad():
        for T in (8, 24, 48, 96, 160):
            cids = torch.randint(0, 47, (1, T))
            cids[0, 0] = 2
            csp = torch.randint(0, 5, (1,))
            crt = torch.ones(1, T)
            cti = torch.zeros(1, T, dtype=torch.long)
            cfm = torch.zeros(1, T, dtype=torch.long)
            cmask = torch.zeros(1, T, dtype=torch.bool)
            cg = _linear_dur(_get(jm, "dur_predictor")(
                cids, csp, cmask, 1.0, None, cti, cfm))
            cg2 = _get(jm, "pitch_predictor")(cids, csp, cmask, cti, cfm)
            _cr, cd = jm(sequence=cids, speaker_ids=csp, sr=48000,
                         symb_durs=None, durs_rate=crt.clone(),
                         pitch_coefs=torch.ones(1, T), device="cpu",
                         gt_durs=cg, gt_pitch=cg2,
                         type_ids=cti, focus_mask=cfm)
            cm = _get(jm, "tacotron")(cids, csp, (cids == 0), cd, cg2,
                                      None)
            l2.append(([voc(cm, 48000, 0.0, True).numpy()],
                       {"mel": cm.numpy()}))
    ok &= stochastic_probe(sess, l2, ["audio"], log.log, atol=3e-3)
    return ok


def parity_l2(jm, sess_acoustic, log):
    """L2 stochastic robustness through the SAME acoustic session."""
    import numpy as np
    import torch

    seed_all()
    cases = []
    with torch.no_grad():
        for idx, T in enumerate((8, 24, 48, 96, 160)):
            cids = torch.randint(0, 47, (1, T))
            cids[0, 0] = 2
            csp = torch.randint(0, 5, (1,))
            crt = torch.ones(1, T)
            cti = torch.full((1, T), 6, dtype=torch.long) if idx % 2 \
                else torch.zeros(1, T, dtype=torch.long)
            cfm = torch.ones(1, T, dtype=torch.long) if idx % 2 \
                else torch.zeros(1, T, dtype=torch.long)
            cmask = torch.zeros(1, T, dtype=torch.bool)
            cg = _linear_dur(_get(jm, "dur_predictor")(
                cids, csp, cmask, 1.0, None, cti, cfm))
            cg2 = _get(jm, "pitch_predictor")(
                cids, csp, cmask, torch.remainder(cti, 6), cfm)
            # Acoustic-level refs: torch main dur + scripted tacotron
            # mel on that (dur, gt_pitch).
            _cr, cd = jm(sequence=cids, speaker_ids=csp, sr=48000,
                         symb_durs=None, durs_rate=crt.clone(),
                         pitch_coefs=torch.ones(1, T), device="cpu",
                         gt_durs=cg, gt_pitch=cg2,
                         type_ids=cti, focus_mask=cfm)
            cm = _get(jm, "tacotron")(cids, csp, (cids == 0), cd, cg2,
                                      None)
            cases.append(([cm.numpy(), cd.numpy()],
                          {"sequence": cids.numpy(),
                           "speaker_ids": csp.numpy(),
                           "durs_rate": crt.numpy(),
                           "gt_durs": cg.numpy(), "gt_pitch": cg2.numpy(),
                           "type_ids": cti.numpy(),
                           "focus_mask": cfm.numpy()}))
    return stochastic_probe(sess_acoustic, cases, ["mel", "dur"], log.log,
                            atol=3e-3)


def parity_e2e(part, out_dir, log, captured):
    """INFO (not a gate): full torch apply_tts vs full ORT chain.

    The chain mirrors torch exactly (recorded ti/fm; dur gt linear;
    pitch gt on base ids). Plain/exclam/focus/emph1 texts must come out
    NEAR-EXACT (torch runs the same full-sequence prediction in-graph).
    emph2 legitimately differs: torch refines gt with windowed
    predictor calls there (multi-sentence focus), the ORT chain uses
    full-sequence gt — the documented engine operating point."""
    import numpy as np
    import onnxruntime as ort
    import torch

    kw = {"providers": ["CPUExecutionProvider"]}
    s_d = ort.InferenceSession(os.path.join(out_dir, FILES["dur_predictor"]),
                               **kw)
    s_p = ort.InferenceSession(
        os.path.join(out_dir, FILES["pitch_predictor"]), **kw)
    s_m = ort.InferenceSession(os.path.join(out_dir, FILES["acoustic"]), **kw)
    s_v = ort.InferenceSession(os.path.join(out_dir, FILES["vocoder"]), **kw)
    for tag, text, feeds, _ta, _td in captured:
        with torch.no_grad():
            t = np.asarray(part.apply_tts(
                text=text, speaker="xenia", sample_rate=48000,
                lang="ru")).ravel()
        seq, sp_ids = feeds["sequence"], feeds["speaker_ids"]
        ti, fm = feeds["type_ids"], feeds["focus_mask"]
        base = torch.remainder(ti, 6)
        feed_d = {"text": seq.numpy(), "speaker_ids": sp_ids.numpy(),
                  "mask": (seq.numpy() == 0),
                  "type_ids": ti.numpy(), "focus_mask": fm.numpy()}
        feed_p = dict(feed_d, type_ids=base.numpy())
        gd = np.exp(s_d.run(None, feed_d)[0].astype(np.float64)).astype(
            np.float32) - 1.0
        gp = s_p.run(None, feed_p)[0]
        mel = s_m.run(None, {"sequence": seq.numpy(),
                             "speaker_ids": sp_ids.numpy(),
                             "durs_rate": feeds["durs_rate"].numpy(),
                             "gt_durs": gd, "gt_pitch": gp,
                             "type_ids": ti.numpy(),
                             "focus_mask": fm.numpy()})[0]
        o = s_v.run(None, {"mel": mel})[0].ravel()
        n = min(len(t), len(o))
        d = float(np.abs(t[:n] - o[:n]).max()) if n else float("inf")
        log.log("e2e-info %s: torch %d vs ort %d samples, max_abs=%.3e"
                % (tag, len(t), len(o), d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination root (filenames from "
                         "MODEL_SETS[%s])" % MODEL_TAG)
    a = ap.parse_args()

    log = Logger()
    begin(log, "export.py (v5_6)",
          "acoustic/dur/pitch/vocoder %s.onnx" % MODEL_TAG)
    log.log("MODEL_SETS[%s] = %s" % (MODEL_TAG, FILES))
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        part, jm = load_source(log)
        captured = capture_feeds(part, jm, log)
        dur, pitch, acoustic, voc, main = transplant_all(jm, log)
        ok = l0_predictors(jm, dur, pitch, log, captured)
        ok &= l0_acoustic(jm, acoustic, log, captured)
        ok &= l0_main(jm, main, log, captured)
        paths = export_all_onnx(dur, pitch, acoustic, voc, a.out_dir,
                                log, captured)
        ok_a, sess_acoustic = parity_acoustic(jm, a.out_dir, log, captured)
        ok &= ok_a
        ok &= parity_predictors(jm, a.out_dir, log, captured)
        ok &= parity_vocoder(jm, a.out_dir, log, captured)
        ok &= parity_l2(jm, sess_acoustic, log)
        parity_e2e(part, a.out_dir, log, captured)
        arts = [paths[k] for k in ("acoustic", "dur_predictor",
                                   "pitch_predictor", "vocoder")]
        finish(log, os.path.join(a.out_dir, "v5_6_export_result.txt"),
               arts, "export(v5_6)", ok)
    except Exception as e:
        import traceback
        log.log("FATAL %s: %s" % (type(e).__name__, str(e)[:500]))
        log.log(traceback.format_exc(limit=5))
        finish(log, os.path.join(a.out_dir, "v5_6_export_result.txt"),
               [], "export(v5_6)", False)


if __name__ == "__main__":
    main()
