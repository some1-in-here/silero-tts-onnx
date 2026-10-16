# scripts/ — one-off ONNX export (build-time only)

This folder produces the five binary assets the runtime needs. It is
**dormant after the first successful build**: daily work (inference,
tests) never touches it — only `silero_onnx/` + `models/onnx/` matter.

## Start here: full build from zero (copy-paste)

Run from the **repo root** (`silero/`). Uses a separate build venv so
the runtime never sees torch:

```bash
cd silero
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate        # source .venv-build/bin/activate on POSIX
pip install -r scripts/requirements-export.txt
# Windows, CPU-only torch (if the default wheel tries CUDA):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# Windows cp1252 console: torch prints ✅/❌ and crashes without UTF-8 mode
# (_common.py also hardens stdout, this is belt-and-suspenders).
# Session-only: deactivate does NOT unset it; it lives until this shell
# closes (harmless — UTF-8 default). Unset manually if needed:
# Remove-Item Env:PYTHONUTF8
$env:PYTHONUTF8 = '1'               # export PYTHONUTF8=1 on POSIX
python scripts/export_all.py        # ~minutes, downloads sources on 1st run
python scripts/check_models.py      # must end with "all 5 models + smoke test passed"
```

That is the whole procedure: `models/onnx/` is now populated and `scripts/`
goes dormant. Details (per-script breakdown, custom `--out-dir`,
expected sizes, gate policy) are below.

## What each script does

| Script | Produces | Source |
|---|---|---|
| `strategies/v5/export_acoustic.py` | `acoustic_v5_ru.onnx` + `.data` | v5 Tacotron side transplanted into `acoustic.py` (via `build_acoustic.py`, 224 tensors), dynamo exporter, opset 18 |
| `strategies/v5/export_vocoder.py` | `vocoder_v5_ru.onnx` | v5 vocoder submodule (`VocWrap`: mel → mag/x/y), `jit.trace` + classic exporter, opset 17 |
| `strategies/v5/export_istft.py` | `istft_48k.onnx` + `.data` | `istft_model.py` (real-valued ISTFT port), dynamo exporter, opset 18 |
| `strategies/v5/export_stress.py` | `silero_stress.onnx` | `silero_stress` homosolver BERT-tiny (TorchScript upstream → classic exporter, opset 17), dynamic B+T |
| `strategies/v5/export_ngram.py` | `ngram_core.onnx` | `silero_stress` EmbeddingBag + 2 MLP heads, `jit.trace` + classic exporter, opset 17 |
| `export_all.py` | v5 (5 models) + v5_6 (4 models) + fixtures, in order | orchestrator: make_fixtures, then `strategies/v5/export.py`, then `strategies/v5_6/export.py`; stops at first failure |
| `make_fixtures.py` | `models/onnx/fixtures/{stress_ref,spec_ref}.pt` | deterministic build inputs: 7 homo windows (vendored tokenizer, torch-free) + real spectrum (torch pipeline, fixed text) |
| `check_models.py` | — | integrity check of `models/onnx/` (names, ORT load, multi-shape robustness, smoke synth) |
| `check_parity.py` | — | end-to-end torch-chain vs ONNX-chain parity (build env only; the `tests/`-forbidden twin of the L1 gates) |
| `_common.py` | — | shared repo-root/out-dir resolution, checker+ORT gates, `.data` externalization |
| `strategies/v5/{acoustic,build_acoustic,encoder,decoder,fftblock,slowmha,predictors,istft_model,layernorm}.py` | — | clean v5 torch ports (mirrors of upstream modules)
| `strategies/v5_6/{export,build,series,forward,vocoder}.py` | — | clean-room v5_6 ports + transplant + L0/L1/L2/e2e gates (reuses v5 blocks, never duplicates) (mirrors of upstream modules; `istft_model.py` is `istft.py` renamed to not clash with `export_istft.py`; `layernorm.py` is an export-safe `LayerNorm` spelling, see `scripts/DEVELOPMENT.md` §2) |

Support modules are imported flat (`from decoder import ...` style):
each script puts its version dir (`strategies/v5/` or `v5_6/`) plus
`scripts/` (`_common`) on `sys.path` via a header bootstrap.

## Build environment setup

Python 3.12, CPU is enough. Use a **separate venv** — the runtime
must never see torch:

```bash
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate        # source .venv-build/bin/activate on POSIX
pip install -r scripts/requirements-export.txt
```

Windows CPU torch (if the default wheel tries CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify: `python -c "import torch, onnx, onnxscript, onnxruntime, silero_stress"`.

(Alternative equivalent: `pip install -e ".[export]" — same extra from
`pyproject.toml`, but default PyPI index, may pull a CUDA torch wheel.
Versions are declared once in `pyproject.toml`; this file only adds the
CPU index.)

## Generating all .onnx files from scratch

Run from the **repo root** (`silero/`):

```bash
cd silero
python scripts/export_all.py              # fixtures + v5 (5 models) + v5_6 (4 models) into models/onnx/
# — or step by step —
python scripts/make_fixtures.py           # models/onnx/fixtures/*.pt (generated refs)
python scripts/strategies/v5/export.py    # all 5 v5_ru models (acoustic/vocoder/istft/stress/ngram)
python scripts/strategies/v5_6/export.py  # all 4 v5_6_ru models (acoustic/dur/pitch/vocoder)
python scripts/check_models.py            # integrity + ORT smoke test
```

Custom destination (must then be passed explicitly everywhere):

```bash
python scripts/export_all.py --out-dir D:\models\silero\onnx
set SILERO_ONNX_MODELS=D:\models\silero\onnx
```

Expected result in `models/onnx/` (~290 MB total, single files, no sidecars):

```
acoustic_v5_ru.onnx                  ~31 MB
vocoder_v5_ru.onnx                   ~58 MB
istft_48k.onnx                       ~92 MB
silero_stress.onnx                   ~117 MB
ngram_core.onnx                      ~8 MB
export_*_result.txt                  per-step gate logs
```

Notes:

- Downloads on first run: `torch.hub` pulls `snakers4/silero-models`
  (acoustic/vocoder/istft sources); `load_accentor()` pulls the stress
  models. Re-runs reuse the torch.hub / HF caches.
- Every script ends with numeric gates (`onnx.checker` + ORT-vs-torch
  `max_abs`; thresholds are printed). A gate failure aborts with
  non-zero exit and the log stays in `export_*_result.txt` — do not
  ship models from a failed run.
- `models/` is git-ignored (see root `.gitignore`): models, gate logs and
  generated fixtures (`models/onnx/fixtures/*.pt`) are built locally, never
  committed. No binaries live in git — not even small ones.

## v5_6 track (isolated)

The `v5_6_ru` model set lives in `scripts/v5_6/` and never touches the
v5 exporters above (shared `_common.py` helpers excepted). File names
come from `silero_onnx.versions.MODEL_SETS["v5_6_ru"]`:

```bash
python scripts/strategies/v5_6/export.py  # 4 models + v5_6_export_result.txt
```
