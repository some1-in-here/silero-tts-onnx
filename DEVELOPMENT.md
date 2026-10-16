# DEVELOPMENT.md — silero-onnx-tts internals

For contributors to this codebase. User-facing usage lives in `README.md`.

## 1. Pipeline architecture

```
text
 │ normalize()            # \s+ collapse, abbreviations (dict + context rules),
 │                        # numbers incl. minus/decimals -> words
 ▼
FullStress                 # homographs first, then plain words:
 │  OnnxStress             #   regex/phrases/dict fast path (pure Python);
 │  │                      #   neural fallback: BERT-tiny -> silero_stress.onnx
 │  │                      #   (input_ids + homo span ids -> variant logits)
 │  └─► OnnxNgram          #   char-ngram featurization (pure Python) ->
 │                         #   ngram_core.onnx (EmbeddingBag + 2 MLP heads) ->
 │                         #   exceptions/user-marks/1-syllable rules (pure Python)
 ▼  stressed text with '+' marks
tokenize() + sos/eos      # 47-symbol table -> ids; speaker name -> embedding id
 │                        # durs_rate / pitch_coefs = ones (plain-text prosody)
 ▼
acoustic_v5_ru.onnx        # ids/spk/rates/coefs -> mel (B,192,F) + durations (B,T)
 ▼
vocoder_v5_ru.onnx         # mel -> mag/x/y spectral parts (B,1201,F)
 ▼
istft_48k.onnx             # (mag,x,y) -> waveform; complex math is real-valued
                           # IDFT (float64 core) inside, no ComplexFloat on boundary
 ▼
WAV 48 kHz mono float32 (soundfile)
```

All five `.onnx` files run in shared `onnxruntime` CPU sessions owned by
`SileroTTS`. The only torch-dependent step in the whole system is *producing*
the models (one-off export, outside this repo).

## 2. Architecture decisions

- **Zero-PyTorch runtime.** No `import torch` in `silero_onnx/` or `tests/`;
  enforced by import assertions in each test file. Rationale: small
  install, no 2 GB dependency, embeddable deployments.
- **Static data ships in the wheel.** `silero_onnx/data/*.json` (homodict,
  phrases, tokenizer vocab, ngrams, exceptions, symbols, speakers, stress
  config) is declared as setuptools `package-data` and verified inside the
  built wheel. Nothing is downloaded at runtime.
- **Generated artifacts go to the user cache.** The BERT tokenizer directory
  (`vocab.txt` + maps, reconstructed from `tokenizer.json`) is written to
  `~/.cache/silero_onnx/tokdir` (`$SILERO_ONNX_CACHE` overrides), never into
  the package — installed wheels are often read-only. A marker file guards
  completeness; content is deterministic.
- **Dynamic shapes, fixed batch.** All ONNX graphs use dynamic `T`/`F` axes;
  batch is specialized to 1 everywhere (loops/pads that require it are
  single-utterance by construction, documented in code).
- **ORT strictness workarounds (all verified, none speculative).**
  Repetition counts are int64 (`Range` rejects mixed int32/int64);
  complex numbers never cross a graph boundary (split real/imag);
  string/dict control flow stays in Python.
- **Error style.** `SystemExit(2/3/4)` with actionable messages for usage /
  environment / inference failures; `ValueError` for domain errors (unknown
  speaker/symbols). No tracebacks for expected failures, no silent wrong output.

## 3. Files in `silero_onnx/`

- `__init__.py` — public facade only: `SileroTTS`, `main`.
- `engine.py` — CLI (`--models/--model-version/--ssml/--examples`),
  console output (colors/examples text). Inference itself lives in
  `strategies/`; model-directory autoresolve for the CLI
  (`SILERO_ONNX_MODELS` env, then `<repo>/models/onnx/`).
- `env.py` — single source of truth for environment variables
  (`MODELS_DIR_VAR`, `CACHE_VAR`, `NO_COLOR_VAR` plus `models_dir()`,
  `cache_dir()`, `colors_enabled()`). Every `os.environ` read in the
  runtime goes through here, tests included — a rename touches one file.
- `versions.py` — `MODEL_SETS` (model filenames per version, single
  source of truth; v5_6 has no `istft` step — measured) +
  `VERSIONS` / `SPEAKERS` + `DEFAULT_VERSION` (`v5_6_ru`).
- `strategies/` — versioned inference behind one contract:
  `base.py` (`TTSStrategy` ABC: abstract `speakers`/`synthesize`,
  concrete `synthesize_to_file()`; the shared vocoder/ISTFT tail serves
  v5 — v5_6 calls its direct vocoder session instead, no `istft`),
  `v5/strategy.py` (`Strategy_V5`: frontend → acoustic → vocoder →
  istft, two-pass `<break>` forcing outside the graph),
  `v5_6/strategy.py` (`Strategy_V5_6`: markers → `_prepare()` feeds
  (`V56Feeds` namedtuple) → dur/pitch predictor sessions →
  `gt_durs=exp(pred)-1`, raw `gt_pitch` → acoustic → vocoder audio;
  SSML rejected loudly, unmatched focus words ignored like upstream),
  `v5_6/emphasis.py` (torch-free port of the upstream v5_6 text
  algorithms: question taxonomy, `**` validation/conversion, focus
  re-marking, sentence spans, `place_tokens`; torch-verified vectors
  live in `tests/test_v56_text.py`),
  `__init__.py` (factory `create_strategy` + `SileroTTS` facade with
  `version=`; unknown versions → `ValueError`, missing assets →
  `SystemExit` listing expected filenames; `model_dir=None` resolves
  through the env var only — the `<repo>/...` fallback is CLI-side).
- `frontend.py` — `normalize()` (whitespace/abbr/numbers), `tokenize()` (+
  `errors="raise"|"filter"`), baked symbol/speaker tables, `TorchFreeFrontend`
  assembling numpy feeds (`prepare()` plain, `prepare_ssml()` with per-token
  prosody arrays and break plans; `text_to_ids()`/`stress_and_tokenize()`
  shared with `Strategy_V5_6`). Unknown chars are *filtered* (matches the
  original pipeline's silent dropping, unlike the strict default of
  `tokenize()`).
- `ssml.py` — stdlib-only `<speak>` parser (break/prosody/p/s, upstream
  tables and errors mirrored, deviations documented); ms→frames clock.
- `homograph_solver.py` — `OnnxStress` (tokenizer + regex/phrases + ORT batch
  + string assembly) and `FullStress` (homographs-then-ngrams ordering with
  the original skip-list wiring). Model/cache dirs resolved via `env`.
- `ngram_accentor.py` — `OnnxNgram`: `<word>` n-grams 1–4 → ids (UNK
  fallback), offsets, ORT core, exceptions/user-marks/1-syllable rules.
  Model dir resolved via `env`.
- `bert_tokenizer.py` — verbatim vendored stdlib-only WordPiece tokenizer.
- `data/` — runtime JSON tables (see above); `tokdir/` is *generated*, not shipped.

## 4. Project structure

```
silero/                          # repo root
├── silero_onnx/                 # torch-free runtime package (ships in the wheel)
│   ├── __init__.py              # public facade: SileroTTS, main
│   ├── engine.py                # CLI, discovery, console output (inference lives in strategies/)
│   ├── versions.py              # MODEL_SETS per version + DEFAULT_VERSION (v5_6_ru)
│   ├── strategies/              # versioned inference behind TTSStrategy + factory
│   │   ├── __init__.py          # create_strategy() + SileroTTS facade (version=)
│   │   ├── base.py              # TTSStrategy ABC (+ shared vocoder/ISTFT tail, v5 path)
│   │   ├── v5/                  # v5 inference strategy package
│   │   │   ├── __init__.py      # re-exports Strategy_V5
│   │   │   └── strategy.py      # Strategy_V5 (pipeline moved unchanged)
│   │   └── v5_6/                # v5_6 inference strategy package
│   │       ├── __init__.py      # re-exports Strategy_V5_6
│   │       ├── strategy.py      # Strategy_V5_6 (markers → predictors → acoustic → vocoder)
│   │       └── emphasis.py      # v5_6 text algorithms port (taxonomy/spans/place_tokens)
│   ├── frontend.py              # normalize/tokenize/TorchFreeFrontend (+ prepare_ssml)
│   ├── ssml.py                  # stdlib-only <speak> parser (break/prosody/p/s tables)
│   ├── homograph_solver.py      # OnnxStress + FullStress (homographs-then-ngrams)
│   ├── ngram_accentor.py        # OnnxNgram (plain-word stress)
│   ├── bert_tokenizer.py        # vendored stdlib-only WordPiece
│   └── data/*.json              # runtime tables shipped as package-data: symbols/speakers/
│                                 # tokenizer/homodict/phrases/ngrams/exceptions/stress_cfg/
│                                 # pause (sil_value for fx_pauses)
├── scripts/                     # one-off build: torch allowed, dormant after build
│   ├── requirements-export.txt  # build-env installer (extra + CPU torch index)
│   ├── export_all.py            # orchestrator: fixtures → v5 (5 models) → v5_6 (4 models)
│   ├── make_fixtures.py         # [gen] onnx/fixtures/*.pt + make_fixtures_result.txt
│   ├── check_models.py          # deployment check (torch-free, generates nothing)
│   ├── check_parity.py          # torch-vs-ONNX e2e parity (build env only, generates nothing)
│   ├── _common.py               # L1/L2 gates, seeds, unified `# header → L1/L2 → RESULT` log format
│   ├── strategies/              # versioned export tracks (torch allowed here, never in runtime)
│   │   ├── v5/                  # everything v5-specific (clean ports + 5 exporters + orchestrator)
│   │   │   ├── export.py        # orchestrator: the 5 v5_ru exporters in order
│   │   │   ├── export_acoustic.py / export_vocoder.py / export_istft.py /
│   │   │   │   export_stress.py / export_ngram.py  # per-model exporters (path bootstrap inside)
│   │   │   └── acoustic.py, build_acoustic.py, encoder.py, decoder.py, fftblock.py,
│   │   │       slowmha.py, predictors.py, istft_model.py, layernorm.py  # clean torch ports
│   │   └── v5_6/                # isolated v5_6_ru export track (reuses v5 blocks, never duplicates)
│   │       ├── export.py        # orchestrator: spy capture → transplant → L0 → dynamo ×4 → L1/L2/e2e
│   │       ├── build.py         # transplants with measured invariants (118 + 112/81 tensors)
│   │       ├── series.py        # SeriesPredictor56 (+type/focus adds) / Dur/PitchPredictor56
│   │       ├── forward.py       # Tacotron56 + Acoustic56 (branch-free control flow)
│   │       ├── vocoder.py       # ConvNeXt backbone + Vocos ISTFT head (RealISTFT reuse)
│   │       └── README.md        # track usage + engine operating point contract
│   ├── README.md                # build quickstart + per-script table
│   └── DEVELOPMENT.md           # exporter internals, thresholds, maintenance
├── tests/                       # torch-free suites (72 tests, no fixtures, no torch)
│   ├── test_production.py       # frontend/stress/facade/synthesis/errors
│   ├── test_extended_robustness.py  # edges/threads/memory/integrity/RTF
│   └── test_ssml.py             # SSML parser contract + prosody/break behavior + CLI
│   ├── test_v56_text.py         # v5_6 text-algorithm vectors (torch-verified)
│   └── test_factory.py          # strategy routing + V56Feeds contract (no models)
├── models/ [gen]                  # local model storage, git-ignored (see models/README.md)
│   ├── onnx/ [gen]                # export pipeline output
│   │   ├── acoustic_v5_ru.onnx [gen]  # ~31 MB single file: ids/spk/rates/coefs -> mel + durations
│   │   ├── vocoder_v5_ru.onnx [gen]   # ~58 MB: mel -> mag/x/y spectral parts
│   │   ├── istft_48k.onnx [gen]       # ~92 MB single file: (mag,x,y) -> waveform (float64 IDFT core)
│   │   ├── silero_stress.onnx [gen]   # ~117 MB homosolver BERT-tiny (input_ids + spans -> logits)
│   │   ├── ngram_core.onnx [gen]      # ~8 MB EmbeddingBag + stress/yo heads
│   │   ├── acoustic_v5_6_ru.onnx [gen]      # ~26 MB: seq/spk/rate/gt/type/focus -> mel + dur
│   │   ├── dur_predictor_v5_6_ru.onnx [gen]      # ~2.5 MB: text/spk/mask/type/focus -> dur
│   │   ├── pitch_predictor_v5_6_ru.onnx [gen]    # ~2.5 MB: text/spk/mask/type/focus -> pitch
│   │   ├── vocoder_v5_6_ru.onnx [gen]            # ~151 MB: mel -> waveform (48 kHz path, no PQMF)
│   │   ├── fixtures/ [gen]            # generated references (make_fixtures.py, seed 42)
│   │   │   ├── stress_ref.pt [gen]    # 7 real homograph windows for export_stress
│   │   │   └── spec_ref.pt [gen]      # real spectrum for export_istft gates
│   │   ├── fixtures_result.txt [gen]        # fixture generation log
│   │   ├── acoustic_export_result.txt [gen] # acoustic L1/L2 gate log
│   │   ├── vocoder_export_result.txt [gen]     # vocoder L1/L2 gate log
│   │   ├── istft_export_result.txt [gen]         # istft L1/L2 gate log
│   │   ├── stress_export_result.txt [gen]       # stress L1/L2 gate log
│   │   ├── ngram_export_result.txt [gen]         # ngram L1/L2 gate log
│   │   └── v5_6_export_result.txt [gen]         # v5_6 L0/L1/L2/e2e gate log
│   └── torch/ [gen]               # raw upstream checkpoints, placed manually
├── pyproject.toml               # runtime deps (torch-free) + export/dev extras
├── README.md                    # user quickstart (zero → audio)
└── DEVELOPMENT.md               # this file
```

Legend: `[gen]` = generated locally by the named script, never committed
(`onnx/`, caches and packaging outputs are git-ignored); everything else
is versioned source. The per-file generator mapping stays maintainable
by convention, not by vigilance: each `export_<name>.py` writes
`onnx/<name>*.onnx`, and `check_models.py::EXPECTED` is the
machine-checked registry of every artifact the build must produce —
adding a model means one row here, one exporter, one `EXPECTED` entry.
v5_6 artifacts come from `scripts/strategies/v5_6/export.py` (names from
`MODEL_SETS`) and are gated by `v5_6_export_result.txt` instead.
v5_6 artifacts come from `scripts/strategies/v5_6/export.py` (names from
`MODEL_SETS`) and are gated by `v5_6_export_result.txt` instead.

## 5. Development principles

- **Evidence over reasoning.** Every component carries a numeric gate
  (bit-exact where achievable, e.g. `max_abs=0.0` on transplants; ≤1e-4
  end-to-end). A passing suite is the definition of done.
- **Separation of concerns.** Neural math (ONNX) vs text rules (Python) vs
  orchestration (`SileroTTS`) never mix in one module.
- **Thread safety.** ORT sessions are shared across threads (documented
  thread-safe for `run()`); frontend state is read-only after init; lazy
  singletons are warmed before fan-out. Covered by a 6-thread test.
- **Cache isolation.** Anything written at runtime goes to tempdirs or the
  user cache; the repo and the package stay read-clean (tests assert behavior,
  and the suite itself writes WAVs only to `tempfile`).
- **No silent degradation.** Ambiguous cases (boundary homographs, .5 duration
  rounding) are pinned as explicit accepted variants, not hidden tolerances.

## 6. Prospects (ordered, each independently shippable)

1. SSML `<break>` on comma positions via the batch-2 support path
   (re-predicted neighbor durations): currently approximated by
   override+silence (pause exact, neighbors may differ slightly).
   Needs a graph re-export for batch-2 duration prediction — do it
   only if the measured delta ever matters. Base `<break>` support
   (two-pass forcing + numpy fx_pauses port) and `<prosody>` are done.
2. 8/24 kHz outputs via the PQMF downsampling path (currently 48 kHz only).
3. GPU execution (CUDA EP) + FP16 revisit: FP16 was measured on CPU and
   rejected (broken loads, destroyed vocoder quality, zero speedup).
4. Batch synthesis (B>1) by generalizing the three single-utterance fast paths.
5. Model regeneration: the export factory lives in `scripts/`
   (`python scripts/export_all.py` from the repo root, needs the torch
   build env); reference fixtures are generated, not versioned
   (`scripts/make_fixtures.py` step 0 → `onnx/fixtures/`, git-ignored).
