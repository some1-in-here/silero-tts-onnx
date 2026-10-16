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
present the graph skips the in-graph prediction, so prosody must be
folded into `gt_pitch` engine-side, and the pitch predictor must get
base type ids (`ti % 6`).
