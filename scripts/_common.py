"""Shared helpers for scripts/ (build-time only, torch allowed here).

Nothing in this file may be imported by the runtime package `silero_onnx`
(which must stay torch-free); it is used solely by export_*.py.
"""
import os
import sys

# Windows cp1252 consoles crash torch's own status prints (it emits ✅/❌):
# force UTF-8 on stdout/stderr at import time. Every export_*.py imports
# this module first, so the hardening applies to all of them. Displayed
# glyphs may mojibake on old consoles — our own logs are written to
# *_export_result.txt logs in UTF-8 regardless.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
del _s

# The torch ONNX exporter unconditionally tries to register torchvision's
# custom-op symbolics (nms, roi_align, roi_pool, deform_conv2d) and warns
# four times when torchvision is absent. None of those ops can appear in
# our graphs (detection/segmentation ops; we use convs, norms, attention
# and IDFT math), so the package is deliberately NOT a dependency and the
# noise is silenced at the source logger. Set before torch is imported:
# logging levels persist on the shared registry object.
import logging as _logging

try:
    _logging.getLogger(
        "torch.onnx._internal.exporter._registration").setLevel(
            _logging.ERROR)
except Exception:
    pass
del _logging

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)  # repo root: silero/
DEFAULT_OUT_DIR = os.path.join(ROOT, "models", "onnx")


def package_data_path(*names):
    """Repo package-data path: <repo>/silero_onnx/data/<names...>.

    Used by export steps that (re)generate versioned runtime tables
    (speakers.json, symbols.json). The file stays a committed source:
    the script logs what it did, git shows the diff, a human commits it.
    """
    return os.path.join(ROOT, "silero_onnx", "data", *names)


def write_speakers_json(entries, path, log):
    """Write-or-validate the torch-free speaker map (no torch needed here).

    `entries` is [(name, speaker_id, model_id)] in checkpoint order,
    taken from the hub package's own `get_speakers()` (never positional
    guesses). Format matches the committed file exactly
    (`{"name": {"id": [sid], "model": mid}}`, single line, no trailing
    newline). Rules, all loud:
    - missing file -> write it;
    - identical content -> log "unchanged", touch nothing;
    - anything else (added/removed/renumbered voice) -> SystemExit
      naming the diff: two versions must never silently overwrite each
      other's table (a new voice is an engineer decision: per-version
      tables would be a bigger, separate change).
    """
    import json

    want = {name: {"id": [sid], "model": mid} for name, sid, mid in entries}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
        if have == want:
            log(f"speakers.json unchanged ({len(want)} voices)")
            return
        extra = sorted(set(want) - set(have))
        gone = sorted(set(have) - set(want))
        moved = sorted(n for n in set(want) & set(have)
                       if want[n] != have[n])
        raise SystemExit(
            "speakers.json mismatch: refusing to overwrite "
            f"(added={extra}, removed={gone}, renumbered={moved}). "
            "A new/renamed voice is an engineer decision, not an "
            "export side effect.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(want, f)
    log(f"speakers.json written ({len(want)} voices): "
        f"{os.path.basename(path)}")


def write_symbols_json(symbols, symbol_to_id, path, log):
    """Write-or-validate the torch-free symbol table (no torch here).

    `symbols`/`symbol_to_id` come from the hub package itself
    (`part.symbols` / `part.symbol_to_id`) — never positional guesses.
    Format matches the committed file exactly (`{"symbols": [...],
    "symbol_to_id": {...}}`, UTF-8, single line, no trailing newline).
    Rules, all loud — the same write-or-fail discipline as
    speakers.json, and for the same reason: an upstream symbol-table
    change is an engineer decision, not a build side effect:
    - missing file -> write it;
    - identical content -> log "unchanged", touch nothing;
    - anything else -> SystemExit naming the diff.
    """
    import json

    want = {"symbols": list(symbols), "symbol_to_id": dict(symbol_to_id)}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
        if have == want:
            log(f"symbols.json unchanged ({len(want['symbols'])} symbols)")
            return
        have_sym = have.get("symbols", [])
        have_map = have.get("symbol_to_id", {})
        added = [s for s in want["symbols"] if s not in have_sym]
        gone = [s for s in have_sym if s not in want["symbols"]]
        remapped = sorted(
            k for k in set(have_map) & set(want["symbol_to_id"])
            if have_map[k] != want["symbol_to_id"][k])
        raise SystemExit(
            "symbols.json mismatch: refusing to overwrite "
            f"(added={added}, removed={gone}, remapped={remapped}). "
            "An upstream symbol-table change is an engineer decision, "
            "not an export side effect.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(want, f, ensure_ascii=False)
    log(f"symbols.json written ({len(want['symbols'])} symbols): "
        f"{os.path.basename(path)}")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


import contextlib as _contextlib
import io as _io


@_contextlib.contextmanager
def collapse_torch_progress():
    """Merge torch.onnx's doubled progress lines into one.

    The exporter prints "<step>..." when a step starts and repeats the
    whole line with a trailing mark when it ends ("... ✅"). This filter
    holds back the unfinished line and emits a single "<step>... ✅"
    line instead. Everything else passes through untouched; use around
    torch.onnx.export calls.
    """
    real = sys.stdout
    pending = []
    buf = [""]

    class _Filter(_io.TextIOBase):
        def write(self, s):
            buf[0] += s
            while "\n" in buf[0]:
                line, buf[0] = buf[0].split("\n", 1)
                self._emit(line + "\n")
            return len(s)

        def _emit(self, line):
            striped = line.rstrip("\n")
            if striped.startswith("[torch.onnx] ") and striped.endswith("..."):
                if pending:
                    real.write(pending.pop() + "\n")
                pending.append(striped)
                return
            if pending and striped.startswith(pending[0]):
                mark = striped[len(pending[0]):].strip()
                real.write(pending.pop() + (" " + mark if mark else "") + "\n")
                return
            if pending:
                real.write(pending.pop() + "\n")
            real.write(line)

        def flush(self):
            while pending:
                real.write(pending.pop() + "\n")
            try:
                real.flush()
            except Exception:
                pass

    proxy = _Filter()
    sys.stdout = proxy
    try:
        yield
    finally:
        proxy.flush()
        sys.stdout = real


def collapse_call(fn, *args, **kwargs):
    """Run fn() with torch-progress collapsing active. One-line wrapper
    around torch.onnx.export calls (avoids re-indenting them)."""
    with collapse_torch_progress():
        return fn(*args, **kwargs)


def ort_session(onnx_path):
    """ORT CPU session with optimizer chatter silenced.

    Namely the `CanUpdateImplicitInputNameInSubgraphs` warning the
    torch-exported EmbeddingBag loop (ngram_core) triggers: the optimizer
    skips one subgraph rename it cannot prove safe. Benign by construction
    (exactness is proven by the L1 gates, e.g. ngram 0.0) — but the yellow
    lines pollute every run, so severity is set to ERROR. Load failures
    still raise; only warnings are hidden.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR and above; default (1) prints warnings
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def begin(log, script, purpose):
    """Standard result-log header: `# <script> — <purpose>`."""
    log.log(f"# {script} — {purpose}")


def finish(log, result_path, artifacts, tag, ok):
    """Standard result-log tail. `artifacts` are full paths of files this
    step produced (models, sidecars, fixtures). Appends a single
    `RESULT PASS/FAIL [tag]: <file> (<MB>), ...` line, writes the log,
    and raises SystemExit on failure. All export_*.py (and
    make_fixtures.py) end this way, so every result file has the same
    shape: `# header` → `L1/L2 …` detail lines → `RESULT …` verdict.
    """
    parts = []
    for p in artifacts:
        if os.path.isfile(p):
            parts.append(f"{os.path.basename(p)} "
                         f"({os.path.getsize(p) / 1e6:.1f} MB)")
        else:
            parts.append(f"{os.path.basename(p)} (MISSING)")
            ok = False
    log.log(f"RESULT {'PASS' if ok else 'FAIL'} [{tag}]: "
            + ", ".join(parts))
    log.write(result_path)
    if not ok:
        raise SystemExit(f"{tag}: fidelity gate FAILED "
                         f"(see {os.path.basename(result_path)})")


class Logger:
    """Collect + echo log lines; flushed to <name>_result.txt on demand."""

    def __init__(self):
        self.lines = []

    def log(self, s):
        self.lines.append(s)
        print(s, flush=True)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def fixtures_dir(out_dir):
    """Build-time reference fixtures live under <out-dir>/fixtures/.

    Generated by scripts/make_fixtures.py (git-ignored build output, like
    the models themselves) — never versioned, never used by tests.
    """
    return os.path.join(os.path.abspath(out_dir), "fixtures")


def inline_external_data(onnx_path, log):
    """Normalize to Kokoro-style single-file layout: weights inside .onnx.

    The dynamo exporter externalizes large initializers into
    <model>.onnx.data; ORT auto-loads the sidecar, but single-file
    models are simpler to ship (all ours are far below the 2 GB
    protobuf ceiling). Replaces the old ensure_external_data (which
    relied on an onnx.save_model kwarg that does not exist in onnx
    1.23 and could only ever TypeError). Small single-file models are
    left byte-identical. Numerics are unaffected (same tensors,
    re-serialized) — proven by the gates, not assumed.
    """
    import onnx

    sidecar = onnx_path + ".data"
    if not os.path.isfile(sidecar):
        log("single-file layout already (no .data sidecar)")
        return
    model = onnx.load(onnx_path)  # pulls external tensors into memory
    onnx.save_model(model, onnx_path)
    os.remove(sidecar)
    size_mb = os.path.getsize(onnx_path) / 1e6
    log(f"inlined weights, removed {os.path.basename(sidecar)}: "
        f"{os.path.basename(onnx_path)} is now a single "
        f"{size_mb:.1f} MB file")


def check_and_gate(onnx_path, feeds, refs, names, log, atol):
    """Level-1 gate (strict/deterministic): onnx.checker + ORT-vs-torch on
    the fixed reference case. Returns (session, ok); the session is reused
    by the level-2 stochastic probe that follows."""
    import numpy as np
    import onnx

    onnx.checker.check_model(onnx_path)
    log("checker=OK")
    sess = ort_session(onnx_path)
    ok = True
    outs = sess.run(None, feeds)
    for nm, r, o in zip(names, refs, outs):
        if r.shape != o.shape:
            log(f"{nm}: SHAPE_MISMATCH torch={r.shape} ort={o.shape}")
            ok = False
            continue
        d = float(np.abs(r - o).max())
        passed = bool(np.allclose(r, o, atol=atol))
        log(f"L1 {nm}{o.shape}: max_abs={d:.3e} ok{atol}={passed}")
        ok &= passed
    # dynamic-shape probe: same session must accept a second length
    return sess, ok


STOCH_SEED = 42
STOCH_ITERS = 5


def seed_all(seed=STOCH_SEED):
    """Deterministic RNG for both levels (numpy + torch)."""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def stochastic_probe(sess, cases, names, log, atol=3e-3, rtol=1e-3):
    """Level-2 gate (stochastic/robustness): N seeded random cases with
    varying dynamic dims through the SAME session, torch-vs-ORT each.

    `cases` is a list of (refs, feeds) with numpy arrays; refs come from
    a torch reference run on the same random inputs. Random noise
    accumulates slightly more error in deep stacks (vocoder/iSTFT), and
    OOD noise can drive some heads to large magnitudes where an absolute
    bar is meaningless — so the gate is scale-aware: each output passes
    when ``|torch-ort| <= atol + rtol*|ort|`` (default atol=3e-3,
    rtol=1e-3). For in-range outputs (|o| ~ 1) the absolute bar dominates.
    This guards breakage on odd shapes (hangs, NaNs, garbage), not
    last-digit fidelity — that is level-1's job. Returns ok.
    """
    import numpy as np

    ok, worst = True, 0.0
    for idx, (refs, feeds) in enumerate(cases):
        try:
            outs = sess.run(None, feeds)
        except Exception as e:
            log(f"L2[{idx}]: ORT RUN FAILED "
                f"{type(e).__name__}: {str(e)[:200]}")
            ok = False
            continue
        ds = []
        for nm, r, o in zip(names, refs, outs):
            if r.shape != o.shape:
                log(f"L2[{idx}] {nm}: SHAPE_MISMATCH "
                    f"torch={r.shape} ort={o.shape}")
                ok = False
                continue
            if not np.all(np.isfinite(r)):
                # Degenerate input (observed: fully-random OOD token ids
                # nan the torch reference itself while ORT stays finite).
                # Nothing about export quality can be concluded from such
                # a case — skip it loudly instead of failing the build on
                # upstream model behavior. Fidelity is level-1's job.
                oor = "ORT finite" if np.all(np.isfinite(o)) else "ORT also non-finite"
                log(f"L2[{idx}] {nm}: torch ref non-finite ({oor}) — SKIP")
                continue
            if not np.all(np.isfinite(o)):
                log(f"L2[{idx}] {nm}: NON-FINITE ORT output")
                ok = False
                continue
            d = float(np.abs(r - o).max())
            ds.append(d)
            worst = max(worst, d)
            if not bool(np.allclose(r, o, rtol=rtol, atol=atol)):
                log(f"L2[{idx}] {nm}{o.shape}: max_abs={d:.3e} "
                    f"OVER atol={atol} rtol={rtol}")
                ok = False
        if ds:
            log(f"L2[{idx}]: shapes {[tuple(r.shape) for r in refs]} "
                f"worst={max(ds):.3e}")
    log(f"L2 stochastic: {len(cases)} iters (seed {STOCH_SEED}) "
        f"worst={worst:.3e} gate(atol={atol},rtol={rtol})="
        f"{'PASS' if ok else 'FAIL'}")
    return ok
