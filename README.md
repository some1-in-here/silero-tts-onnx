<!-- Maintainers: README is user-facing only — no internals, no how-it-works,
     no numeric measurements. Documentation rules and the doc map: CONTRIBUTING.md. -->

# silero-onnx-tts 🔊

A PyTorch-independent Russian text-to-speech engine powered by Silero TTS
and ONNX Runtime. Text in, WAV out — no `torch` in the runtime process.

Two model generations: **v5_ru** and **v5_6_ru**(the default).

## Quickstart: from zero to audible speech (copy-paste)

```bash
cd silero

# 1. Build the ONNX models into models/onnx/ (one-time setup; requires torch).
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate   # plain `python` uses the venv's Python 3.12; `py -3.12` would bypass it
python -m pip install -r scripts/requirements-export.txt   # torch + onnx/onnxscript stacks, CPU-only torch index
$env:PYTHONUTF8 = '1'   # Windows only: torch prints ✅/❌; cp1252 can crash without UTF-8 mode
                        # Session-only: persists after deactivate until the shell closes; unset with `Remove-Item Env:PYTHONUTF8`
python scripts/export_all.py
python scripts/check_models.py   # expect "all 9 models + smoke tests (v5_ru + v5_6_ru) passed"
deactivate  # leave the build venv; you can delete .venv-build afterward

# 2. Install the runtime (no torch).
py -3.12 -m pip install -e .   # onnxruntime, numpy, soundfile, num2words only

# 3. Generate speech (models/onnx/ is discovered automatically).
silero-onnx "Привет! Меня зовут Ксения." -o out.wav --speaker xenia
```

Play `out.wav` in any audio player, or stream speech directly to your speakers:

```python
import sounddevice as sd  # pip install sounddevice
from silero_onnx import SileroTTS

tts = SileroTTS(model_dir="path/to/models")  # explicit path (or SILERO_ONNX_MODELS); v5_6_ru by default
audio, _ = tts.synthesize("Меня зовут **Ксения**.", speaker="xenia")
sd.play(audio, 48000)
sd.wait()
```

**Next:** [Install](#install) · [Usage](#usage)

Build details: [`scripts/README.md`](scripts/README.md). Development and internals: [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Features

- ✨ Emphasis (v5_6_ru): `**word**` emphasized, `*word*` focused,
  `<intensity=0..3>` focus strength — see [Emphasis](#emphasis-v5_6_ru-only).
- 🗣️ 5 voices: `aidar`, `baya`, `kseniya`, `eugene`, `xenia`, 48 kHz, CPU.
- 📦 Runtime deps only: `onnxruntime`, `numpy`, `soundfile`, `num2words`.
- ⚡ Fast on CPU: ~25× realtime — no GPU needed.
- 🧠 Smart frontend: abbreviations, numbers (including negative and fractional),
  automated stress, yo-letter and homograph disambiguation.
- 🎯 Faithful output: validated against the original Silero pipeline by the
  test suite.

## Install

Two environments, strictly separated: torch (~2 GB) lives **only** in a
throwaway build venv used to export the models. Everyday use needs just the
lean runtime. If you download prebuilt models from Releases, you can skip the
build environment entirely.

| | 🏭 Build env | 📦 Runtime env |
|---|---|---|
| What | the factory (model exporters) | the runtime (this package) |
| torch | yes (export needs it) | never (stays torch-free) |
| Where | a separate `.venv-build` | your normal environment |
| When | once (to build `models/onnx/`, then dormant) | always (to use the voices) |

### 🏭 Build env (only to export or regenerate models)

```bash
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate   # plain `python` uses the venv's Python 3.12; `py -3.12` would bypass it
python -m pip install -r scripts/requirements-export.txt   # torch + onnx/onnxscript stacks, CPU-only torch index
$env:PYTHONUTF8 = '1'   # Windows only: torch prints ✅/❌; cp1252 can crash without UTF-8 mode
                        # Session-only: persists after deactivate until the shell closes; unset with `Remove-Item Env:PYTHONUTF8`
python scripts/export_all.py
python scripts/check_models.py   # expect "all 9 models + smoke tests (v5_ru + v5_6_ru) passed"
deactivate  # leave the build venv; you can delete .venv-build afterward
```

### 📦 Runtime env (everyday use, no torch)

```bash
py -3.12 -m pip install -e .   # onnxruntime, numpy, soundfile, num2words only
```

## Models

The ONNX models are data, not code: they are **not** bundled in the wheel.
Download the ready-made files from
[GitHub Releases](https://github.com/some1-in-here/silero-tts-onnx/releases)
— one ZIP with everything, or individual files — into a local directory.

| Model set | Files | Approx. size |
|---|---|---|
| `v5_6_ru` (default) | 4 own (acoustic, duration & pitch predictors, vocoder) + 2 shared (stress, ngram) | ~290 MB |
| `v5_ru` | 3 own (acoustic, vocoder, istft) + the same 2 shared (stress, ngram) | ~290 MB |
| both sets (one ZIP) | all 9 | ~470 MB |

Point the runtime at that directory with `SILERO_ONNX_MODELS` (or the
`model_dir=` argument). The CLI also discovers `<repo>/models/onnx/`
automatically.

No prebuilt release for your case? Build the models once:
[`scripts/README.md`](scripts/README.md).

## Usage

Two interfaces, one engine. Select the model version with
`version="v5_ru"` in Python or `--model-version v5_ru` in the CLI.
The default is `v5_6_ru` (emphasis markers); both versions do SSML,
with one difference: `<prosody pitch>` works on `v5_ru` only.

### A. Python library (embed in your code)

```python
from silero_onnx import SileroTTS

tts = SileroTTS(model_dir="path/to/models")  # or SILERO_ONNX_MODELS; defaults to v5_6_ru
audio, _ = tts.synthesize("Меня зовут **Ксения**.", speaker="xenia")  # float32 mono, 48 kHz
tts.synthesize_to_file("Привет, мир!", "out.wav", speaker="baya")
print(tts.speakers)  # ['aidar', 'baya', 'eugene', 'kseniya', 'xenia']

tts5 = SileroTTS(model_dir="path/to/models", version="v5_ru")  # classic v5 pipeline
```

### B. CLI utility (one-liners, scripts, no Python code)

```bash
silero-onnx "Сначала с нажимом: **важно**. Потом без нажима: важно." -o emph.wav
silero-onnx --model-version v5_ru "Привет! Меня зовут Ксения." -o out5.wav

silero-onnx --help      # --models-dir, --output, --speaker, --model-version, --ssml
silero-onnx --examples  # runnable samples: versions, emphasis, stress, SSML
silero-onnx --speaker-list   # needs no models; --model-list lists versions
```

One utterance per call: for very long texts, split at sentence boundaries
(same ~1000-character guidance as the original Silero).

## Emphasis (v5_6_ru only)

| Markup | Effect | Example |
|---|---|---|
| `**word**` | Emphasized | `Меня зовут **Ксения**.` |
| `*word*` | Focused | `Меня зовут *Ксения*.` |
| `<intensity=N>` | Scales the strength of `*`/`**` marks, 0–3 (default 3) | `<intensity=1>*Тихо* говорю.</intensity>` |

`<intensity=N>` only modulates words that already carry a `*` / `**`
mark; without marks it has no effect — a warning is printed (the
original Silero behaves the same). The level is per-utterance: if
several tags disagree, the first one wins and a warning is printed.
`**...**` wraps a single word — spaces inside are invalid markup and
are rejected exactly like upstream (`**два** **слова**`, not
`**два слова**`). `*...*` may span several words: each word of the
span becomes focused, exactly as upstream (verified bit-identical).
In SSML the tag is stripped before XML parsing, exactly as upstream
does; plain text accepts it too.

## SSML (`--ssml` / `ssml=True`)

Both model versions support SSML. The `<speak>` root is mandatory: without
it, the input is rejected with a one-line error (no traceback). Plain text
without `--ssml` never parses tags: `<break/>` would be vocalized as words.
User `+` stress marks keep working inside SSML, and on `v5_6_ru` the
`**` / `*` emphasis markers do too.

| Tag | What it does | Example |
|---|---|---|
| `<break time="500ms"/>` / `<break time="2s"/>` | Exact pause (milliseconds or whole seconds) | `…пауза. <break time='500ms'/> Вот…` |
| `<break strength="…"/>` | `x-weak` 25 ms … `x-strong` 1000 ms | `<break strength='weak'/>` |
| `<prosody rate="…">` | `x-slow` 0.5 … `x-fast` 1.5, or `150%` | `<prosody rate='fast'>Скорее!</prosody>` |
| `<prosody pitch="…">` | `x-low` 0.75 … `x-high` 1.25, `robot` 0, or `+12%` / `-10%` — `v5_ru` only | `<prosody pitch='high'>Высоко.</prosody>` |
| `<p>` / `<s>` | Paragraph/sentence breaks (`x-strong` / `strong`) | `<s>Первое.</s><s>Второе.</s>` |

`<prosody pitch>` on `v5_6_ru` warns (once per utterance) and is ignored:
that model takes pitch from its own predictor, rate and breaks work
normally. A break on a phrase-final position can be shorter than requested
— the sentence-final duration cap wins there, exactly as in the original
Silero models.

Run `silero-onnx --examples` for runnable samples.

## Environment variables

| Variable | What it does |
|---|---|
| `SILERO_ONNX_MODELS` | model directory (validated by the version's acoustic file). Priority: explicit `model_dir=` argument → this var → `<repo>/models/onnx` autodiscovery (**CLI only**; library code never looks there) |
| `SILERO_ONNX_CACHE` | tokenizer cache root (default `~/.cache/silero_onnx`) |
| `NO_COLOR` | disable CLI colors (standard convention; colors are also auto-off on pipes) |
| `PYTHONUTF8` | not ours — stock Python: forces UTF-8 output (needed on cp1252 Windows consoles) |

Set per-session (`$env:SILERO_ONNX_MODELS='D:\models\silero\onnx'` /
`export SILERO_ONNX_MODELS=...`) or persistently (Windows environment
settings / `.bashrc`). Nothing is stored in the project — env vars live
in the OS environment, each process reads them at startup.

## Tests

```bash
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest tests/ -v
```

Model-dependent tests need a built `models/onnx/` directory (or
`SILERO_ONNX_MODELS`); without it they skip automatically with an explicit
reason. Development and internals: [`DEVELOPMENT.md`](DEVELOPMENT.md);
contribution and documentation rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

This project is licensed under **CC BY-NC-SA 4.0** (see `LICENSE`).
In short: non-commercial use only, credit the authors, share alike.

- Original author: the **Silero Team**
  ([silero-models](https://github.com/snakers4/silero-models)).
- This product is a **derivative work** (Adapted Material) based on
  Silero TTS v5, used under CC BY-NC-SA 4.0. What was changed:
  weights transplanted into reimplemented modules and exported to
  ONNX (acoustic/vocoder/ISTFT), stress/ngram cores exported to ONNX,
  frontend and tooling rewritten torch-free.
- Stress/accentor models come from
  [`silero-stress`](https://github.com/snakers4/silero-stress) under the MIT license,
  © Silero Team.