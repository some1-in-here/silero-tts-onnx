<!-- Audience: contributors and AI agents; internals, measurements, gates and
     CI rationale live here. Keep README.md strictly user-facing — rules and
     the doc map: CONTRIBUTING.md. -->

# DEVELOPMENT.md — silero-onnx-tts internals

For contributors to this codebase. User-facing usage lives in `README.md`;
project-wide contribution and documentation rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).

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

The runtime has nine unique `.onnx` files: the v5_ru set has 5 entries,
the v5_6_ru set has 6 (4 own + the same 2 shared `silero_stress` /
`ngram_core`). They are opened through the shared `ort.py` session factory;
each `TTSEngine` owns the subset for its selected version and is exposed
through the `SileroTTS` facade. The runtime itself has no torch-dependent
step; producing and validating the models is a one-off build task via
`scripts/` (see `scripts/README.md`).

`v5_6_ru` (the default) shares the frontend but replaces the tail:
predictor sessions feed durs/pitch plus the SSML `pitch_coefs` into
`acoustic_v5_6_ru.onnx`, and the fused `vocoder_v5_6_ru.onnx` maps mel
straight to audio (no istft step). The diagram above shows the v5_ru
route.

## 2. Architecture decisions

- **Zero-PyTorch runtime.** No `import torch` in `silero_onnx/` or `tests/`;
  enforced by import assertions in each test file plus a static AST scan
  of the package (`tests/test_production.py`). Rationale: small
  install, no 2 GB dependency, embeddable deployments.
- **Layered factories, injected dependencies.** `EngineFactory` resolves
  `(language_id, version_id)` through `LanguageRegistry`/`ModelRegistry`,
  then injects the language into the frontend and the frontend/sessions/
  version into the strategy. Adding a language or a model version is a
  new class/spec + registration entry, not an edit to existing classes.
- **Explicit capability gates.** A language declares `requires_stress`;
  a version declares `supports_stress`/`supports_ssml`/`supports_pitch`/
  `supports_questions`.
  Incompatible pairs fail in `EngineFactory` before model I/O; per-call
  `ssml=True` on a non-SSML version raises `ValueError`; a no-stress
  version receives a zero focus mask and its language's `apply_stress`
  is a no-op. No silent `AttributeError`/`KeyError` at inference time.
- **Static data ships in the wheel.** `silero_onnx/data/*.json` (homodict,
  phrases, tokenizer vocab, ngrams, exceptions, symbols, speakers, stress
  config) is declared as setuptools `package-data`, so it is included in
  an installed wheel/package; nothing is downloaded at runtime.
- **Generated artifacts go to the user cache.** The BERT tokenizer directory
  (`vocab.txt` + maps, reconstructed from `tokenizer.json`) is written to
  `~/.cache/silero_onnx/tokdir` (`$SILERO_ONNX_CACHE` overrides), never into
  the package — installed wheels are often read-only. A marker file guards
  completeness; content is deterministic.
- **Dynamic shapes, fixed TTS batch.** The acoustic/vocoder/predictor/iSTFT
  graphs are single-utterance by construction (batch 1, dynamic sequence /
  frequency axes as applicable);
  the stress/ngram frontend graphs additionally expose dynamic batch/word
  axes (`B`/`T` for stress, `K`/`W` for ngram; documented in
  `scripts/DEVELOPMENT.md`).
- **ORT strictness workarounds (all verified, none speculative).**
  Repetition counts are int64 (`Range` rejects mixed int32/int64);
  complex numbers never cross a graph boundary (split real/imag);
  string/dict control flow stays in Python.
- **Error style.** `SystemExit` with actionable messages for usage /
  environment / inference failures; `ValueError` for domain errors (unknown
  speaker/symbols). No tracebacks for expected failures, no silent wrong output.

### Architectural layers

| Слой | Директория | Классы | Ответственность |
|------|-----------|--------|-----------------|
| Язык | `silero_onnx/languages/` | `AbstractLanguage`, `Language_Ru`, `LanguageRegistry` | Текстовые правила, ударения, символы, спикеры. Не знает о версиях моделей. |
| Фронтенд | `silero_onnx/frontend/` | `AbstractTextFrontend`, `TextFrontend`, `TextFrontend_Ru`, `TextPipeline` | Текст → входы модели. Язык внедряется через DI. Не знает о версиях моделей. |
| Версии моделей | `silero_onnx/models/` | `ModelSpec`, `ModelSet`, `Version_V5Ru`, `Version_V56`, `ModelRegistry` | Наборы ONNX-файлов, топология сессий, акустические входы, флаги возможностей. |
| Общая логика | `silero_onnx/strategies/common/` | `FeedBuilder`, `BreakProcessor`, `PauseService` | Версия-независимый SSML lowering, паузы, сборка фидов. |
| Нейтральный SSML | `silero_onnx/common/` | `SsmlLoweringService` | SSML lowering вне `strategies/`, чтобы фронтенд не зависел от стратегий. |
| Стратегии | `silero_onnx/strategies/` | `AbstractTTSStrategy`, `Strategy_V5`, `Strategy_V56` | Версия-специфичная последовательность сессий и сборка фидов. |
| Оркестратор | `silero_onnx/engine.py` | `EngineFactory`, `TTSEngine`, `SileroTTS` | Создаёт движок из пары (язык, версия модели). |
| Экспорт | `scripts/strategies/` | `AbstractExportPipeline`, `ExportPipeline_V5`, `ExportPipeline_V56` | Template Method для ONNX-экспорта по версиям. |

## 3. Files in `silero_onnx/`

- `__init__.py` — public facade only: `SileroTTS`, `EngineFactory`, `main`.
- `engine.py` — orchestrator + CLI. `EngineFactory.create_engine(
  language_id, version_id, model_dir)` resolves both registries, rejects
  incompatible language/version capability pairs and versions without a
  registered strategy *before* touching model files, opens the version
  sessions, and injects language/frontend/sessions/version into the
  strategy. The frontend itself is selected through `FrontendRegistry`
  (language id -> frontend factory, generic `TextFrontend` fallback), so a
  new language does not add an `if language_id == ...` branch here.
  CLI (`--output/--speaker/--models-dir/
  --model-version/--ssml/--examples/--speaker-list/--model-list`), console output, and
  CLI-only model-dir autoresolve
  (`SILERO_ONNX_MODELS`, then `<repo>/models/onnx/`) also live here.
- `cli.py` — thin re-export of `engine.main` (console-script target).
- `env.py` — single source of truth for environment-variable names
  (`MODELS_DIR_VAR`, `CACHE_VAR`, `NO_COLOR_VAR`) and the runtime accessors
  `models_dir()`, `cache_dir()`, `colors_enabled()`. Runtime reads go
  through here; tests use the same name constants, so a rename touches one
  place.
- `versions.py` — backward-compatible projection of the model layer:
  `MODEL_SETS`, `VERSIONS`, `DEFAULT_VERSION` (`v5_6_ru`). New code
  imports `silero_onnx.models`.
- `models/` — declarative model-version layer (knows nothing about
  languages/frontends):
  `spec.py` (`ModelSet`, `ModelSpec`: ONNX files, session topology,
  acoustic input names, `supports_stress`/`supports_ssml`/`supports_pitch`/
  `supports_questions`, pause fill), `versions.py` (`Version_V5Ru`,
  `Version_V56`), `registry.py` (`ModelRegistry`).
- `languages/` — language abstraction independent of model versions:
  `base.py` (`AbstractLanguage`: `normalize`, `apply_stress`,
  `get_symbol_table`, `get_speakers`, `requires_stress`; no-stress
  `apply_stress` is a no-op), `registry.py` (`LanguageRegistry`), and
  `ru/` (`Language_Ru` + abbreviations/numbers/stress/symbols).
- `frontend/` — text frontend with the language injected via DI:
  `base.py` (`AbstractTextFrontend`), `pipeline.py` (`TextPipeline`),
  `stages.py` (`NormalizeStage`, `AbbrevStage`, `NumberStage`,
  `StressStage`, `TokenizeStage`), `registry.py` (`FrontendRegistry`),
  `ru.py` (`TextFrontend_Ru`; the legacy `TorchFreeFrontend` name remains
  an alias). The frontend receives the version's `supports_pitch`
  capability from the factory and emits `pitch_coefs` only when it is
  true. `__init__.py` re-exports the old module-level
  `normalize()`/`tokenize()`/table helpers so existing callers keep
  working.
- `common/` - version-neutral services shared by the frontend and
  strategies: `ssml_lowering.py` (`SsmlLoweringService`, `SsmlPlan`,
  `SegmentPlan`: the single parse→normalize→segment merge→rate/pitch
  attribution algorithm; it imports no
  model-version or ONNX-session code, so the frontend can depend on it
  directly).
- `strategies/` - version-specific inference behind one contract:
  `base.py` (`AbstractTTSStrategy`: request validation including the
  version's `supports_ssml` gate, unknown-speaker checks, shared
  `synthesize_to_file()`; v5's shared vocoder/ISTFT tail),
  `common/` (`BreakProcessor`/`PauseService`: two-pass rate forcing +
  numpy `fx_pauses` with a model-specific silence fill; `FeedBuilder`:
  version-input assembly,
  `durs_rate` finite-and-`> 0` validation, zero focus masks for no-stress
  versions; `ssml_lowering.py` is a backward-compatibility re-export only,
  while strategies import the canonical `common/` module directly),
  `pauses.py` (compatibility re-export of `common/breaks.py`),
  `v5/strategy.py` (`Strategy_V5`), `v5_6/strategy.py` (`Strategy_V56`,
  historical alias `Strategy_V5_6`: markers + SSML rate/pitch/breaks →
  predictors → acoustic → vocoder; pitch_coefs drive the checkpoint's
  update_pitch_coef on the predictor pitch; `<intensity=N>` is stripped
  pre-XML like upstream and scales focus masks), `v5_6/emphasis.py` (torch-free
  port of the upstream v5_6 text algorithms; vectors in
  `tests/test_v56_text.py`), `__init__.py` (`StrategyFactory`; lazy
  `create_strategy`/`SileroTTS` compatibility hooks; `Feeds_V56`
  canonical feed namedtuple with legacy `V56Feeds` alias).
- `ssml.py` — stdlib-only `<speak>` parser (break/prosody/p/s; upstream
  tables are mirrored, with deliberate error-policy deviations documented
  in the module). Unsupported tags and unknown `<break>`/`<prosody>`
  attributes raise `ValueError` instead of being silently dropped.
  Provides the ms→frames clock. Version-independent lowering lives only
  in `common/ssml_lowering.py`; strategies import it directly from there,
  and the old `strategies/common/ssml_lowering.py` path is a thin
  compatibility re-export.
- `ort.py` — single runtime session factory: 2 intra-op threads, 1
  inter-op thread, sequential execution (reproducible CPU policy). Used
  by acoustic/vocoder/predictor sessions and by stress/ngram inference.
- `homograph_solver.py` — `OnnxStress` (tokenizer + regex/phrases + ORT
  batch + string assembly) and `FullStress` (homographs-then-ngrams
  ordering with the original skip-list wiring). Model/cache dirs via
  `env`.
- `ngram_accentor.py` — `OnnxNgram`: `<word>` n-grams 1–4 → ids (UNK
  fallback), offsets, ORT core, exceptions/user-marks/1-syllable rules.
- `bert_tokenizer.py` — verbatim vendored stdlib-only WordPiece tokenizer.
- `data/` — runtime JSON tables (see above); `tokdir/` is *generated*,
  not shipped.

## 4. Project structure

```
silero/                          # repo root
├── silero_onnx/                 # torch-free runtime package (ships in the wheel)
│   ├── __init__.py              # public facade: SileroTTS, EngineFactory, main
│   ├── engine.py                # EngineFactory/orchestrator + CLI + console output
│   ├── cli.py                   # console-script target (re-export of engine.main)
│   ├── env.py                   # env vars single source of truth
│   ├── versions.py              # compat MODEL_SETS / VERSIONS / DEFAULT_VERSION
│   ├── ort.py                   # shared ORT session policy: 2 intra, 1 inter, sequential
│   ├── models/                  # declarative model-version layer
│   │   ├── spec.py              # ModelSet / ModelSpec (files, topology, capabilities)
│   │   ├── versions.py          # Version_V5Ru / Version_V56
│   │   └── registry.py          # ModelRegistry
│   ├── languages/               # language layer (no model versions)
│   │   ├── base.py              # AbstractLanguage (+ requires_stress)
│   │   ├── registry.py          # LanguageRegistry
│   │   └── ru/                  # Language_Ru + abbreviations/numbers/stress/symbols
│   ├── frontend/                # text frontend (language injected via DI)
│   │   ├── base.py              # AbstractTextFrontend
│   │   ├── pipeline.py          # TextPipeline
│   │   ├── registry.py          # FrontendRegistry
│   │   ├── stages.py            # Normalize/Abbrev/Number/Stress/Tokenize stages
│   │   └── ru.py                # TextFrontend_Ru (+ legacy TorchFreeFrontend)
│   ├── common/                  # version-neutral services shared by frontend + strategies
│   │   └── ssml_lowering.py      # SsmlLoweringService/SsmlPlan/SegmentPlan
│   ├── strategies/              # versioned inference behind AbstractTTSStrategy
│   │   ├── __init__.py          # StrategyFactory + compat create_strategy/SileroTTS
│   │   ├── base.py              # AbstractTTSStrategy (validation + shared tail)
│   │   ├── common/              # breaks.py, feeds.py; ssml_lowering.py compat re-export
│   │   ├── pauses.py            # compat re-export of common/breaks.py
│   │   ├── v5/strategy.py       # Strategy_V5
│   │   └── v5_6/                # Strategy_V56 (+ historical Strategy_V5_6), emphasis.py
│   ├── ssml.py                  # stdlib-only <speak> parser (break/prosody/p/s tables)
│   ├── homograph_solver.py      # OnnxStress + FullStress (homographs-then-ngrams)
│   ├── ngram_accentor.py        # OnnxNgram (plain-word stress)
│   ├── bert_tokenizer.py        # vendored stdlib-only WordPiece
│   └── data/*.json              # runtime tables shipped as package-data: symbols/speakers/
│                                 # tokenizer/homodict/phrases/ngrams/exceptions/stress_cfg/
│                                 # pause (sil_value for fx_pauses)
├── scripts/                     # one-off build: torch allowed, dormant after build
│   ├── requirements-export.txt  # build-env installer (extra + CPU torch index)
│   ├── export_all.py            # orchestrator: fixtures → v5 (5 new) → v5_6 (4 new; 2 shared)
│   ├── make_fixtures.py         # [gen] fixtures + speakers/symbols.json validation + result log
│   ├── check_models.py          # deployment check (torch-free, generates nothing)
│   ├── check_parity.py          # torch-vs-ONNX e2e parity (build env only, generates nothing)
│   ├── package_models.py        # release ZIP packer (file list derived from MODEL_SETS)
│   ├── _common.py               # L1/L2 gates, seeds, unified `# header → gates → RESULT` log format
│   ├── strategies/              # versioned export tracks (torch allowed here, never in runtime)
│   │   ├── base.py              # AbstractExportPipeline (prepare→export→validate→finalize)
│   │   ├── v5/                  # everything v5-specific
│   │   │   ├── export.py        # ExportPipeline_V5: the 5 exporters in order
│   │   │   ├── export_acoustic.py / export_vocoder.py / export_istft.py /
│   │   │   │   export_stress.py / export_ngram.py  # per-model exporters (path bootstrap inside)
│   │   │   ├── build.py         # canonical build import (build_acoustic)
│   │   │   ├── vocoder.py       # canonical vocoder-export import
│   │   │   ├── stress.py        # canonical stress-export import
│   │   │   └── acoustic.py, build_acoustic.py, encoder.py, decoder.py, fftblock.py,
│   │   │       slowmha.py, predictors.py, istft_model.py, layernorm.py  # clean torch ports
│   │   └── v5_6/                # isolated v5_6_ru export track (reuses v5 blocks)
│   │       ├── export.py        # ExportPipeline_V56: spy capture → transplant → L0 → dynamo ×4 → L1/L2 + e2e gate
│   │       ├── acoustic.py      # Tacotron_V56 + Acoustic_V56 (branch-free control flow)
│   │       ├── dur_predictor.py / pitch_predictor.py  # canonical predictor imports
│   │       ├── build.py         # transplants with per-tensor shape asserts + measured invariants
│   │       ├── series.py        # SeriesPredictor_V56 (+type/focus adds) / Dur/PitchPredictor_V56
│   │       ├── forward.py       # Main_V56 fused parity port; re-exports Acoustic_V56/Tacotron_V56
│   │       ├── vocoder.py       # ConvNeXt backbone + Vocos ISTFT head (RealISTFT reuse)
│   │       └── README.md        # track usage + engine operating point contract
│   ├── README.md                # build quickstart + per-script table
│   └── DEVELOPMENT.md           # exporter internals, thresholds, maintenance
├── tests/                       # torch-free suites (131 tests; model tests skip without models)
│   ├── conftest.py              # model-set detection + requires_models/requires_v56 skips
│   ├── test_production.py       # frontend/stress/facade/synthesis/errors + static torch-import scan
│   ├── test_extended_robustness.py  # edges/threads/memory/integrity/RTF
│   ├── test_ssml.py             # SSML parser contract + lowering + prosody/break/intensity behavior + CLI
│   ├── test_v56_text.py         # v5_6 text-algorithm vectors (torch-verified)
│   ├── test_factory.py          # strategy routing + Feeds_V56 contract (no models)
│   └── test_architecture.py     # language/version/strategy/factory capability contracts (no models)
├── models/ [gen]                  # local model storage, git-ignored (build output only)
│   ├── onnx/ [gen]                # export pipeline output
│   │   ├── acoustic_v5_ru.onnx [gen]  # ~31 MB single file: ids/spk/rates/coefs -> mel + durations
│   │   ├── vocoder_v5_ru.onnx [gen]   # ~58 MB: mel -> mag/x/y spectral parts
│   │   ├── istft_48k.onnx [gen]       # ~92 MB single file: (mag,x,y) -> waveform (float64 IDFT core)
│   │   ├── silero_stress.onnx [gen]   # ~117 MB homosolver BERT-tiny (input_ids + spans -> logits)
│   │   ├── ngram_core.onnx [gen]      # ~8 MB EmbeddingBag + stress/yo heads
│   │   ├── acoustic_v5_6_ru.onnx [gen]      # ~26 MB: seq/spk/rate/gt/pitch_coefs/type/focus -> mel + dur
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
│   │   └── v5_6_export_result.txt [gen]         # v5_6 L0/L1/L2 + e2e gate log
│   └── torch/ [gen]               # raw upstream checkpoints, placed manually
├── pyproject.toml               # runtime deps (torch-free) + export/dev extras
├── README.md                    # user quickstart (zero → audio)
├── CONTRIBUTING.md              # contribution + documentation rules (doc map)
└── DEVELOPMENT.md               # this file
```

Legend: `[gen]` = generated locally by the named script, never committed
(`models/`, caches and packaging outputs are git-ignored); everything
else is versioned source. The per-file generator mapping stays
maintainable by convention, not by vigilance: each version's exporter
(v5's per-model `export_<name>.py` steps or the v5_6 `export.py`) writes
its artifacts into `models/onnx/`, and `check_models.py::EXPECTED`
holds the I/O contract of every artifact while enforcing full coverage
of `MODEL_SETS` (both the v5_ru set and the v5_6_ru set). Acoustic input
names come from `ModelSpec.acoustic_inputs()` rather than a duplicated
list. Adding a model means one `MODEL_SPECS` entry (`MODEL_SETS` is
derived from it), one exporter registration (the v5 `STEPS` list or the
v5_6 `artifacts`/`export_all_onnx`), and one non-acoustic `EXPECTED` row
(the acoustic input contract comes from `ModelSpec`); a new version track
also gets an `export_all.py` step.
`scripts/package_models.py` derives the release ZIP contents from the
same `MODEL_SETS`, so a release cannot silently omit a model set.

## 5. CI workflows

Three GitHub Actions workflows, split by cost:

| Workflow | Trigger | What it runs |
|---|---|---|
| `ci.yml` | every push (any branch) + pull requests + manual (`workflow_dispatch`) | torch-free, model-free: runtime + dev extras, torch-free import and light CLI smoke checks, `pytest` — model-dependent tests skip and are reported in one summary line (per-test reasons available with `-rs`; see `tests/conftest.py`) |
| `e2e.yml` | every push to `master` (default branch; merges included) + manual (`workflow_dispatch`; pick any branch in the Run workflow dialog) | full pipeline: build env → `export_all.py` (both model sets) → `check_models.py` (all 9 artifacts) → `check_parity.py` (v5 e2e; the v5_6 e2e gate runs inside its export step) → full `pytest --require-models` → `package_models.py` dry-run; gate logs summarized and uploaded |
| `export_and_release.yml` | `v*` tags (or manual dispatch) | same export/validate/test chain, then packages and publishes the GitHub Release (ZIP + individual models) |

Model tests carry the `requires_models` (v5_ru set) or `requires_v56`
(v5_6_ru set) marker; the E2E and release runs always have the models
and pass `--require-models`, so a missing set fails the job there
instead of skipping. Short per-workflow summary and the run
instructions live next to the files:
[`.github/workflows/README.md`](.github/workflows/README.md).

## 6. Development principles

- **Evidence over reasoning.** Every exported model carries a numeric gate
  (bit-exact where achievable, e.g. `max_abs=0.0` on the iSTFT/ngram
  exports; the end-to-end waveform gate is `5e-4`, with wider diagnostic
  mel/mag bars documented in `scripts/DEVELOPMENT.md`). A passing suite is
  the definition of done.
- **Separation of concerns.** Neural math (ONNX) vs text rules (Python) vs
  orchestration (`EngineFactory`/`TTSEngine`, exposed by `SileroTTS`)
  never mix in one module.
- **Thread safety.** ORT sessions are shared across threads (documented
  thread-safe for `run()`); frontend state is read-only after init; lazy
  singletons are warmed before fan-out. Covered by a 6-thread test.
- **Cache isolation.** Anything written at runtime goes to tempdirs or the
  user cache; the repo and the package stay read-clean (tests assert behavior,
  and the suite itself writes WAVs only to `tempfile`).
- **No silent degradation.** Ambiguous cases (boundary homographs, .5 duration
  rounding) are pinned as explicit accepted variants, not hidden tolerances.
- **Layered documentation.** `README.md` is for users only (setup, usage,
  examples); "how it works", measurements and CI details live here or in
  `scripts/*`. Contract and doc map: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 7. Prospects (ordered, each independently shippable)

1. v5_ru SSML `<break>` on comma positions via the batch-2 support path
   (re-predicted neighbor durations): currently approximated by
   override+silence (pause exact, neighbors may differ slightly).
   Needs a graph re-export for batch-2 duration prediction — do it
   only if the measured delta ever matters. Base `<break>` support
   (two-pass forcing + numpy fx_pauses port, shared by both tracks) and
   `<prosody>` are done; v5_6_ru never hits this path upstream either
   (its always-gt operating point skips the support branch).
2. 8/24 kHz outputs via the PQMF downsampling path (currently 48 kHz only).
3. GPU execution (CUDA EP) + FP16 revisit: FP16 was measured on CPU and
   rejected (broken loads, destroyed vocoder quality, zero speedup).
4. Batch synthesis (B>1) by generalizing the three single-utterance fast paths.
5. Model regeneration: the export factory lives in `scripts/`
   (`python scripts/export_all.py` from the repo root, needs the torch
   build env); reference fixtures are generated, not versioned
   (`scripts/make_fixtures.py` step 0 → `models/onnx/fixtures/`, git-ignored).
