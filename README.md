# silero-onnx-tts 🔊

A PyTorch-independent Russian text-to-speech engine powered by Silero TTS
and ONNX Runtime. Text in, WAV out — no `torch` in the runtime process
(verified by the test suite via `sys.modules`).

Two model generations: **v5_ru** (classic pipeline, SSML-capable) and
**v5_6_ru** (emphasis markers, the default).

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
python scripts/check_models.py   # expect "all models + smoke test passed"
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
- ⚡ Fast: RTF ~25x on CPU (4.15 s of audio in ~0.18 s wall time) — on par
  with torch CPU on the same texts, while dropping the ~2 GB torch
  dependency entirely.
- 🧠 Smart frontend: abbreviations, numbers (including negative and fractional),
  automated stress, yo-letter and homograph disambiguation.
- 📏 Measured quality vs the original pipeline: max absolute waveform error
  ~1e-4 for v5 (9.4e-05 torch-chain vs ONNX-chain); single-span v5_6
  emphasis matches at ~2e-04 end to end.

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
python scripts/check_models.py   # expect "all models + smoke test passed"
deactivate  # leave the build venv; you can delete .venv-build afterward
```

### 📦 Runtime env (everyday use, no torch)

```bash
py -3.12 -m pip install -e .   # onnxruntime, numpy, soundfile, num2words only
```

## Usage

Two interfaces, one engine. Select the model version with
`version="v5_ru"` in Python or `--model-version v5_ru` in the CLI.
The default is `v5_6_ru`.

### A. Python library (embed in your code)

```python
from silero_onnx import SileroTTS

tts = SileroTTS(model_dir="path/to/models")  # or SILERO_ONNX_MODELS; defaults to v5_6_ru
audio, _ = tts.synthesize("Меня зовут **Ксения**.", speaker="xenia")  # float32 mono, 48 kHz
tts.synthesize_to_file("Привет, мир!", "out.wav", speaker="baya")
print(tts.speakers)  # ['aidar', 'baya', 'eugene', 'kseniya', 'xenia']

tts5 = SileroTTS(model_dir="path/to/models", version="v5_ru")  # classic pipeline + SSML
```

### B. CLI utility (one-liners, scripts, no Python code)

```bash
silero-onnx "Меня зовут **Ксения**." -o out.wav --speaker xenia
silero-onnx --model-version v5_ru "Привет! Меня зовут Ксения." -o out5.wav

silero-onnx --help      # --models-dir, --output, --speaker, --model-version, --ssml
silero-onnx --examples  # runnable samples: versions, emphasis, stress, SSML
silero-onnx --speaker-list   # needs no models; --model-list lists versions
```

## Emphasis (v5_6_ru only) ✨

| Markup | Effect | Example |
|---|---|---|
| `**word**` | Emphasized | `Меня зовут **Ксения**.` |
| `*word*` | Focused | `Меня зовут *Ксения*.` |
| `<intensity=N>` | Focus strength 0–3 (default 3) | `<intensity=1>Тихо говорю.</intensity>` |

Single-span emphasis matches the original pipeline near-exactly.
Multi-sentence focus uses full-sequence predictor passes by design
(the engine's operating point — see [`scripts/strategies/v5_6/README.md`](scripts/strategies/v5_6/README.md)).

## SSML (`--ssml` / `ssml=True`, v5_ru only)

The `<speak>` root is mandatory. Without it, the input is rejected with a
one-line error (no traceback). Plain text without `--ssml` never parses
tags: `<break/>` would be vocalized as words.

| Tag | What it does | Example |
|---|---|---|
| `<break time="500ms"/>` / `<break time="2s"/>` | Exact pause (milliseconds or whole seconds) | `…пауза. <break time='500ms'/> Вот…` |
| `<break strength="…"/>` | `x-weak` 25 ms … `x-strong` 1000 ms | `<break strength='weak'/>` |
| `<prosody rate="…">` | `x-slow` 0.5 … `x-fast` 1.5, or `150%` | `<prosody rate='fast'>Скорее!</prosody>` |
| `<prosody pitch="…">` | `x-low` 0.75 … `x-high` 1.25, `robot` 0, or `+12%` / `-10%` | `<prosody pitch='high'>Высоко.</prosody>` |
| `<p>` / `<s>` | Paragraph/sentence breaks (`x-strong` / `strong`) | `<s>Первое.</s><s>Второе.</s>` |

How it works: prosody becomes per-token rate/pitch arrays (one acoustic
pass, no model changes). `<break>` pauses force exact durations
(500 ms → 40 frames) in a second pass plus mel silencing. User `+`
stress marks continue to work inside SSML.

Limits, all measured: fractional seconds (`1.5s`) fail like upstream;
breaks on commas skip upstream's support re-prediction (the pause
itself stays exact). Run `silero-onnx --examples` for runnable samples.

## Tests

```bash
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest tests/ -v  # 72 tests, ~25 s, torch-free guaranteed
```

## Models

Inference needs a directory with self-contained ONNX models (single
files, weights included — no `.data` sidecars).

**v5_ru** (5 files):

- `acoustic_v5_ru.onnx` (~31 MB) — text + speaker + prosody → mel + durations
- `vocoder_v5_ru.onnx` (~58 MB) — mel → split spectral parts (no complex boundary)
- `istft_48k.onnx` (~92 MB) — spectrum → waveform (real-valued ISTFT inside)
- `silero_stress.onnx` (~117 MB) — homograph disambiguation (BERT-tiny)
- `ngram_core.onnx` (~8 MB) — plain-word stress + yo

**v5_6_ru** (4 files + the shared two above):

`acoustic_v5_6_ru.onnx` (~26 MB),
`dur_predictor_v5_6_ru.onnx` + `pitch_predictor_v5_6_ru.onnx`
(~2.5 MB each), `vocoder_v5_6_ru.onnx` (~151 MB, direct mel → audio).

These are produced once by the export pipeline (see [`DEVELOPMENT.md`](DEVELOPMENT.md)).
They are data, not code, and are not part of the wheel. Pass the directory
explicitly or set `SILERO_ONNX_MODELS`.

**Shortcut without a build:** download the `.onnx` files from the
[GitHub Releases](https://github.com/some1-in-here/silero-tts-onnx/releases)
page into a local `models/onnx/` directory. Then `pip install -e .` is all the
runtime needs.

## Notes

- Single utterance per call (batch = 1); very long inputs follow the
  original ~1000-symbol guidance.
- v5_6_ru takes plain text with markers (no SSML); v5_ru takes plain
  text or SSML.

## Environment variables

| Variable | What it does |
|---|---|
| `SILERO_ONNX_MODELS` | model directory (validated by the version's acoustic file). Priority: explicit `model_dir=` argument → this var → `<repo>/onnx` autodiscovery (**CLI only**; library code never looks there) |
| `SILERO_ONNX_CACHE` | tokenizer cache root (default `~/.cache/silero_onnx`) |
| `NO_COLOR` | disable CLI colors (standard convention; colors are also auto-off on pipes) |
| `PYTHONUTF8` | not ours — stock Python: forces UTF-8 output (needed on cp1252 Windows consoles) |

Set per-session (`$env:SILERO_ONNX_MODELS='D:\models\silero\onnx'` /
`export SILERO_ONNX_MODELS=...`) or persistently (Windows environment
settings / `.bashrc`). Nothing is stored in the project — env vars live
in the OS environment, each process reads them at startup.

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