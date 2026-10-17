# scripts/strategies/v5_6/ — isolated v5_6_ru export track

Everything needed to build the `v5_6_ru` model set lives here and never
imports from the v5 exporters in `scripts/` (shared helpers in
`scripts/_common.py` excepted: logging, gates, single-file layout).

## Contents

- `export.py` — the only orchestrator: loads `v5_6_ru` from torch.hub,
  captures exact torch feeds with a spy proxy, exports 4 single-file
  models via TS2EPConverter + dynamo (vocoder via trace, like v5),
  then runs L1/L2/emphasis parity gates. Output paths come exclusively
  from `silero_onnx.versions.MODEL_SETS["v5_6_ru"]`.

## Usage

From the repo root, with the build env installed:

```console
pip install -r scripts/requirements-export.txt
python scripts/strategies/v5_6/export.py [--out-dir onnx]
```

Result: `acoustic_v5_6_ru.onnx`, `dur_predictor_v5_6_ru.onnx`,
`pitch_predictor_v5_6_ru.onnx`, `vocoder_v5_6_ru.onnx` plus
`v5_6_export_result.txt` (unified `# header → L1/L2 → RESULT` format).

## Key deviation from v5 (measured, see export.py docstring)

`acoustic_v5_6_ru.onnx` is text→*mel* (+dur), `vocoder_v5_6_ru.onnx` is
mel→audio — the v5 session topology, split at the mel boundary (a real
intermediate of the torch graph). New inputs on the acoustic side:
`type_ids`, `focus_mask`, `gt_durs`/`gt_pitch` (`pitch_coefs` has no
input — the graph ignores it under gt). The engine operating point is
*always-gt* (gt from the standalone predictor sessions); with gt
present the graph skips the in-graph prediction, so pitch prosody would
have to be folded into `gt_pitch` engine-side — the runtime warns and
ignores pitch tags instead (see below) — and the pitch predictor must
get base type ids (`ti % 6`).

## Runtime SSML support (engine-side, no export changes)

- `<prosody rate>` fills the existing `durs_rate` input; the graph
  divides by it after the gt clamp, so in-graph rate semantics are
  preserved with no graph changes.
- `<break>` replays upstream's in-graph `dur[sid] = sdur` override via
  a two-pass `durs_rate` rewrite: pass 2 rewrites only the break
  token's rate so `round(base/rate)` hits the requested frame count,
  and the graph's own sos/tail caps then apply exactly as they do after
  upstream's override (verified frame-exact against the scripted
  forward). The mel pause region is filled by the numpy `fx_pauses`
  port with this set's `sil_value` (`-5.0`, the `_PAUSE_SIL_VALUE`
  constant in `silero_onnx/strategies/v5_6/strategy.py`, dumped from
  the checkpoint; v5_ru uses `-8.0` from `data/pause.json`).
- Comma breaks take the same override+silence path: upstream's batch-2
  support re-prediction only runs when `gt_durs` is absent, which the
  always-gt operating point never is (measured in the scripted
  forward).
- `<prosody pitch>` is warned about and ignored: under always-gt the
  export has no `pitch_coefs` input (above), and `gt_pitch` is the raw
  predictor output the gates validate.
- `<intensity=N>` is stripped before XML parsing (upstream's
  `extract_intensity_from_ssml` order) and scales the `*focus*` mask;
  the first tag wins for the whole utterance (differing values warn);
  with no `*`/`**` marks it is inert (upstream audio is identical) and
  warns loudly.
