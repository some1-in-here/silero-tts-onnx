# Fixed-voice multilingual exports

This additive path supports the ten checkpoint identities listed in
`scripts/multilingual-checkpoints.json`. It does not change the existing
Russian `SileroTTS` factory, CLI, SSML, emphasis or stress strategies.

## Use

Install the normal runtime dependencies. Obtain an exported model directory
containing `frontend.json`, `acoustic.onnx`, `vocoder.onnx`, and, for Vocos
models, `istft.onnx`. Graphs are separate release assets, not source files or
wheel contents. Do not point this API at the Russian five/six-stage bundles.

```python
import soundfile as sf
from silero_onnx.multilingual import MultilingualRuntime

tts = MultilingualRuntime("models/multilingual/v3_en", lower_memory=True)
audio, symbol_durations = tts.synthesize("Hello, world!", "en_0", "en")
sf.write("hello.wav", audio, tts.data["sample_rate"])
```

Use a fixed voice from `frontend.json`'s `speaker_to_id`. The return value is
mono floating-point audio plus model symbol durations, not streaming chunks.
Only 48 kHz was exported and validated. This API is synchronous and should
be serialized by its host; cancellation/playback belongs to the host.

`lower_memory=True` disables ONNX Runtime CPU arenas and memory patterns. It
reduces retained process memory at a throughput cost; it does not quantize
weights. Both memory modes were compared against identical input. Benchmark
numbers in the evaluation report are complete-utterance return times, not
first audible audio or a latency promise for another computer.

## Text requirements and limitations

- Plain text only. No SSML, random voices, cloning or automatic stress.
- Numbers and isolated letter names need host-side expansion. Numeric-only
  text can be rejected by the original frontend; do not silently drop a
  failed chunk. The host should report it or use its own fallback.
- V4 Indic expects the checkpoint's ISO-romanized input, not native script.
  No romanizer or its dependencies are bundled. The evaluation used
  Aksharamukha 2.3 separately (AGPL-3.0); hosts must assess that dependency
  separately. This API does not perform transliteration automatically.
- CIS extended models require correctly supplied word stress. The base
  no-stress model still expects explicit stress for Slavic languages per
  Silero's instructions. Russian stress modules are not reused blindly.
- Maximum cleaned text is 1000 model symbols. Split longer text before
  inference. This is an export bound, not a recommendation for long screen
  reader chunks; the host may require much shorter chunks.
- Supported-script filtering mirrors the checkpoint rather than adding
  language detection or translations. Unsupported-only input raises an
  explicit error. Native pronunciation and broad text coverage have not
  been certified by this contribution's author.

Sonata's ongoing number/letter processing is separate work. A reusable
external library may follow once validated; it is not included or required
by this contribution.

## Rebuild

Use the repository's separate export environment. Checkpoints are executable
Python packages: obtain them from the official URLs in the inventory and
verify their SHA-256. Never load an untrusted `.pt` file.

```text
python scripts/export_multilingual.py v3_en --checkpoint models/torch/v3_en.pt --out-dir models/multilingual/v3_en
```

The output directory must be empty. The command refuses unreviewed checkpoint
hashes, compares the pure frontend with the original, checks reconstructed
acoustic/vocoder math against the complete original output, checks ONNX
structure, and gates waveform length, finite values, durations and maximum
absolute waveform difference (below 0.005 on normalized floating-point
audio). It writes paired reference/converted WAVs and `export-parity.json`.
Those generated files are deliberately ignored by Git.

Exporter diagnostic metadata (including local source-file paths) is removed
before the final ONNX/audio gates. This does not modify graph operators,
initializers, precision, shapes or model calculations.

Recheck a completed export in the separate torch-free runtime environment:

```text
python scripts/check_multilingual.py --model-dir models/multilingual/v3_en --fixtures-dir models/multilingual/v3_en --lower-memory
python scripts/package_multilingual.py --model-dir models/multilingual/v3_en --output dist/silero-v3_en-onnx-48000.zip
```

Packaging checks graph hashes and uses a strict file allowlist. It includes
the model's license and attribution, not executable source checkpoints,
recordings, logs or Python code. It refuses to replace an existing ZIP.

Keep `scripts/multilingual-samples.json` provenance with test results. Several
Western/Indic test sentences are machine-authored, not native-reviewed.
Some published voices/languages have no representative test sentence in this
set. A successful export is not a universal language-quality verdict.

## Port details

Build-time ports reuse the existing V5 transformer modules, preserve FP32
neural weights, and copy the original learned SnakeBeta activations and
PQMF reconstruction rather than substituting another activation. Spectral
V5 models retain the FP64 real inverse FFT. The runtime applies BASIC
optimization only to iSTFT: ORT 1.23 otherwise creates an unsupported FP64
FusedMatMul. Acoustic/vocoder optimization remains at the default level.

Original model weights and architecture: Silero Team. Starting ONNX ports:
Vladimir Govorov / some1-in-here. Multilingual adaptation, Windows CPU
evaluation and NVDA integration: Jim Grimsby Jr., with AI assistance. See
`MULTILINGUAL-NOTICES.md` for component-specific terms and
`docs/multilingual-evaluation.md` for the boundaries of the evidence.
