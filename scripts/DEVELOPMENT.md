# scripts/DEVELOPMENT.md — how the exporters work

Maintainer notes for `scripts/`. Runtime internals live in the root
`DEVELOPMENT.md`; user build steps live in `scripts/README.md`.

## 1. Tracing strategy per model (PyTorch → ONNX)

**Acoustic (`export_acoustic.py`, dynamo exporter, opset 18).**
The upstream v5 graph is not exported directly: `build_acoustic.py`
transplants 224 tensors from the JIT Tacotron into `acoustic.py`
(`AcousticClean`: encoder + dur/pitch predictors + speaker embedding +
length regulator + HourGlass decoder + lin). The transplant count is
asserted (`== 224`) — any upstream weight rename breaks loudly here,
not silently in audio. Exported with `torch.onnx.export(..., dynamo=True)`
and symbolic T (batch fixed to 1): the classic torchscript exporter can
no longer lower this graph on current torch (layer_norm/add collapse to
rank-unknown `onnx::` nodes, breaking the transposes), and the module
is dynamo-ready by construction (branch-free clamps, `layernorm.py`,
reshape seam, no `.item()`). Two pre-gates run before export: torch-vs-pipeline
durations (`max_abs <= 1e-4`, observed `0.0`) — mel is compared against the transplant
proof, not against a zeros-pitch tacotron-direct call (whose `pitch_proj`
bias makes it an invalid reference).

**Vocoder (`export_vocoder.py`, classic exporter, opset 17).**
The JIT vocoder is wrapped (`VocWrap`) to freeze
`(mel, 48000, 0.0, True)` into constants, leaving a single `mel`
input (B,192,T). Traced on a REAL mel from the clean acoustic (fixed
T32 text — deterministic; synthetic randn made the gate flaky).
Straight export; gate is ORT-vs-torch `atol=1e-3` (measured
2.3e-04..7.7e-04 on mag across real + seeded inputs: deep-conv
kernel noise between torch/ORT BLAS sits systematically above 1e-4
on current stacks).

**iSTFT (`export_istft.py`, dynamo exporter, opset 18).**
`istft_model.py::RealISTFT` re-implements `torch.fft.irfft` + overlap-add
+ window-normalization with real tensors only (see §2). It is first
proven against the genuine `wrapped_jit_v.istft` on a generated real
spectrum (`models/onnx/fixtures/spec_ref.pt` via `make_fixtures.py`) and
a synthetic T=9 case (`atol=1e-4`, else exit 2 without exporting), then
dynamo-exported with dynamic T.

**Stress (`export_stress.py`, classic exporter, opset 17).**
`load_accentor().homosolver.model` (BERT-tiny + span head) arrives as a
TorchScript module, which dynamo refuses — so the classic path is used
(the natural fit for scripted graphs), with `dynamic_axes` B(1..32)/T(1..512).
Reference: 7 real homograph windows (generated fixture
`models/onnx/fixtures/stress_ref.pt` via `make_fixtures.py`). Gates:
`max_abs <= 1e-4` *and* variant agreement — sigmoid-round must select the
same homograph variant in torch and ORT (logit closeness alone could hide
a `.5`-boundary flip), plus a B=3/T=13 second-shape probe through the
same session.

**Ngram (`export_ngram.py`, classic exporter, opset 17).**
`NgramCore` re-hosts the accentor's `embedding` (as `EmbeddingBag(…,16,
mean)`), `stress_clf`, `yo_clf`. Character n-gram featurization
(`<word>`, n=1..4, dict hit, UNK=126521, cumulative offsets) is *not*
exported — strings cannot cross the boundary — it stays in
`silero_onnx/ngram_accentor.py::featurize` (bit-identical copy inside
the exporter for the reference run). Traced on 10 featurized words;
gate `atol=1e-4` on both heads.

## 2. Working around ONNX Runtime strictness

All five graphs run under ORT CPU with far stricter op semantics than
eager torch. The workarounds below are load-bearing — removing any of
them breaks export or inference (each was a measured failure, not a guess):

- **No complex tensors on any boundary** (`vocoder`, `istft`).
  `spec = mag*(x + i·y)` is split into three real inputs/outputs.
  The IDFT core mirrors the Hermitian spectrum explicitly
  (`xr_full`/`xi_full` in `istft_model.py::_irfft`) and keeps only the
  real part. Upstream alternative `exp_vocoder.py` (complex `spec`
  input) was prototyped and rejected — kept out of `scripts/` on purpose.
- **float64 IDFT core** (`istft_model.py::_irfft`).
  The 2400-term accumulation in float32 measured ~1e-4 error; casting
  the `(B,T,n)` projections to double keeps fidelity at ~1e-7. The rest
  of the graph (and all boundaries) stays float32.
- **`aten::format` stub** (`export_acoustic.py`).
  The JIT graph formats debug strings; the classic exporter has no
  symbol for it. Registered symbolic replaces it with a constant —
  verified tensor-neutral (string-only dataflow).
- **Shape inference off for acoustic** (`GLOBALS.onnx_shape_inference =
  False`). The length-regulator tiling is data-dependent; the inference
  pass mis-folds it (verified export failure). ORT performs its own
  shape propagation at load.
- **int64 repetition/pad counts.** ORT `Range`/tiling rejects mixed
  int32/int64; all dynamic counts (`featurize` offsets, span ids,
  `Sequence`/`speaker_ids`) are int64 on both sides.
- **torchvision is absent by design, not by accident.** The exporter
  probes for its custom-op symbolics (`nms`, `roi_align`, `roi_pool`,
  `deform_conv2d` — detection/segmentation ops) and warns when missing.
  None can occur in our graphs, so it stays out of the build env and the
  four warnings are silenced in `_common.py` (source logger to ERROR).
- **LayerNorm spelled with primitives** (`scripts/strategies/v5/layernorm.py`).
  Current torch rewrites `aten::layer_norm` to `onnx::LayerNormalization`
  mid-export, dropping the output rank; the following `transpose(1,2)`
  then fails ("transpose for tensor of unknown rank", torch 2.14).
  `ExportLayerNorm` computes the identical formula (biased variance,
  eps=1e-5) from mean/var/normalize ops, which are shape-preserving.
  Verified stack: torch 2.14.1 + onnx 1.23.2 + onnxscript 0.7.2.
- **Dynamic axes, batch = 1.** All graphs export `T`/`F`/`K`/`W`/`N`
  dynamic; batch-dependent control flow (punkt loop, padding) is valid
  only for single utterances — the runtime pins batch 1 by construction.
- **Single-file models (no sidecars).** The dynamo exporter
  externalizes large initializers into `.onnx.data`, but all our models
  are far below the 2 GB protobuf ceiling — so `_common.
  inline_external_data` folds weights back into each `.onnx` right
  after export (single files like Kokoro/Supertonic; proven bit-exact
  at 0.0 vs the sidecar layout). The runtime loads either layout, but
  the factory always ships single files.
- **FP16 rejected.** Measured on CPU: broken loads, destroyed vocoder
  quality, zero speedup. Graphs stay FP32 (see root DEVELOPMENT.md §7).

## 3. Maintaining the exporters (upstream updates)

1. **Upstream moved (snakers4/silero-models, silero-stress).**
   Re-run `export_all.py`; the first failing gate names the drift:
   transplant assert (weight rename) → update `build_acoustic.py` +
   the mirror module; ISTFT param assert (n_fft/hop/win) → re-base
   `istft_model.py`; new input arity → extend `input_names` +
   `TorchFreeFrontend.prepare`.
2. **`silero_onnx/data/*.json` regeneration is out of scope here.**
   The JSON tables ship in the wheel and are versioned; exporters only
   rebuild `.onnx` weights. If upstream retrains (new vocab/homodict),
   the dumps must be re-taken from live objects — a separate procedure.
3. **Threshold policy (three levels: L0/L1/L2).**
   L0 same-math transplant proofs (clean predictors and acoustic port vs
   the scripted originals, torch-vs-torch): absolute bars `5e-6`
   (predictors) and `3e-5` (acoustic mel). The float32 kernel floor is
   not bit-stable across runners: the v0.4.0 release run on
   ubuntu-latest measured pitch/plain2 `1.013e-6` (just over the old
   `1e-6` bar) and acoustic mel/plain2 `8.583e-6` (86% of the old `1e-5`
   bar), while earlier runs of the same commit stayed under both. Bars
   keep ~3.5-5x headroom over the worst observation — still 20x/3.3x
   under the L1 regression bars, and a real transplant bug (wrong
   weights/op wiring) is orders of magnitude larger.
   L1 strict (deterministic, fixed reference): `1e-4` torch-vs-torch/ORT
   (`1e-3` for acoustic mel/dur and vocoder mag/x/y, whose reduction-order
   and BLAS-kernel noise is larger — measured, not guessed). Never widen
   an L1 gate to green a run: a breach means graph divergence. The same
   rule applies to L0: widen only with documented cross-platform
   measurements showing floor-level noise, never to hide a breach.
   L2 stochastic (5 seeded cases, seed 42, varying T/B/W): relaxed
   (`3e-3` acoustic/vocoder, `1e-3` elsewhere) — random noise accumulates
   more error in deep stacks, so L2 guards breakage (hangs, NaNs,
   shape mismatches), never last-digit fidelity. That is L1's job.
   The L2 comparison is scale-aware (`|d| <= atol + 1e-3*|o|`): OOD
   noise can saturate heads to large magnitudes where a pure absolute
   bar is meaningless; in-range outputs are still gated by `atol`.
   Cases whose torch reference itself goes non-finite on degenerate
   noise (observed with fully-random token ids) are SKIPped loudly —
   no export conclusion is possible there, and failing the build on
   upstream model behavior would be dishonest.
   Boundary `.5` cases (duration rounding, `мел/мёл` variants) are
   pinned as accepted multi-variants in tests, not hidden in exporters.
   End-to-end (`check_parity.py`, the `tests/`-forbidden twin) has one
   extra axis: torch-vs-ONNX noise is platform-dependent even on the
   identical package stack (long-text mag 3.8e-3 on Windows vs 5.9e-3
   on GitHub ubuntu-latest). Its waveform bar stays tight (5e-4;
   measured <=2.8e-4 on both), while mel/mag bars are diagnostics with
   deliberate headroom above the worst observation (2e-2) - a runner
   change must not flake a diagnostic, and a broken export is orders of
   magnitude off, not ~1.5x. Never widen the waveform bar to green a run.
4. **Adding a model.** Copy the smallest fitting exporter, keep the
   pattern: argparse `--out-dir` → `begin()` header → reference inputs
   (real, not random, wherever content matters) → pre-gate → export →
   `checker` → L1 ORT-vs-torch gate → second-shape probe → L2
   `stochastic_probe` (`_common.py`, seed 42) → `finish()` RESULT tail
   (unified `# header` → `L1/L2` → `RESULT PASS/FAIL` format) →
   non-zero exit on failure. Register it in `export_all.py`, in the
   `scripts/README.md` table, in the root `DEVELOPMENT.md` map, and in
   `check_models.py::EXPECTED` (+ a `_robustness` block there:
   torch-free numpy multi-shape runs).
