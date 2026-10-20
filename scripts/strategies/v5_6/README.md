# scripts/strategies/v5_6/ — isolated v5_6_ru export track

Everything specific to the `v5_6_ru` export track lives here. It reuses
the proven v5 block implementations and shared helpers in
`scripts/_common.py` (logging, gates, single-file layout), but never
imports the v5 exporters in `scripts/`.

## Contents

- `export.py` — the only orchestrator (`ExportPipeline_V56`, Template
  Method): loads `v5_6_ru` from torch.hub, captures exact torch feeds with
  a spy proxy, transplants weights into the clean eager ports, exports 4
  single-file models with dynamo (opset 18; the vocoder is exported from
  its clean port, not a traced TorchScript submodule), then runs L0/L1/L2
  gates plus a hard e2e parity gate (`GATE_E2E = 2e-2`) that includes
  `pitch_high` / `pitch_low` / `pitch_robot` SSML cases. Output paths
  come exclusively from
  `silero_onnx.versions.MODEL_SETS["v5_6_ru"]`.
  `emph2` is the one captured case where torch supplies a windowed,
  refined gt; the gate feeds that recorded gt into ORT and compares it
  with the captured torch main using the effective pitch (the scripted
  gt path skips `update_pitch_coef`, while the always-gt port must apply
  it). The other cases compare against fresh `apply_tts` calls. The
  runtime still uses full-sequence predictors for every utterance, and
  those e2e cases exercise that operating point.
- `acoustic.py` - `Tacotron_V56` + `Acoustic_V56` (branch-free acoustic
  half, including the checkpoint's `update_pitch_coef` channel).
- `dur_predictor.py` / `pitch_predictor.py` - canonical predictor imports
  (`DurPredictor_V56`, `PitchPredictor_V56` from `series.py`).
- `build.py` - weight transplant with measured invariants.
- `series.py` - Transformer series predictors (+type/focus adds).
- `forward.py` - `Main_V56` fused parity port; re-exports
  `Acoustic_V56`/`Tacotron_V56`.
- `vocoder.py` — ConvNeXt backbone + Vocos ISTFT head.
- `../base.py` — shared `AbstractExportPipeline`
  (prepare → export → validate/session-open → finalize).

## Usage

From the repo root, with the build env installed:

```console
pip install -r scripts/requirements-export.txt
python scripts/strategies/v5_6/export.py [--out-dir onnx]
```

Result: `acoustic_v5_6_ru.onnx`, `dur_predictor_v5_6_ru.onnx`,
`pitch_predictor_v5_6_ru.onnx`, `vocoder_v5_6_ru.onnx` plus
`v5_6_export_result.txt` (unified `# header → L0/L1/L2 + e2e gate → RESULT` format).

## Key deviation from v5 (measured, see export.py docstring)

`acoustic_v5_6_ru.onnx` is text→*mel* (+dur), `vocoder_v5_6_ru.onnx` is
mel→audio — the v5 session topology, split at the mel boundary (a real
intermediate of the torch graph). New inputs on the acoustic side:
`type_ids`, `focus_mask`, `gt_durs`, `gt_pitch`, `pitch_coefs`. The
engine operating point is *always-gt* (gt from the standalone predictor
sessions); with gt present the scripted graph skips in-graph prediction
and ignores `pitch_coefs`, but upstream's normal `apply_tts` path feeds
no gt and does apply `update_pitch_coef`. The clean acoustic port
therefore applies that same transform to the raw predictor pitch, so
`<prosody pitch>` keeps upstream semantics under always-gt. The pitch
predictor must get base type ids (`ti % 6`).

## SSML handling (engine-side + acoustic pitch channel)

- `<prosody rate>` fills the existing `durs_rate` input; the graph
  divides by it after the gt clamp, so in-graph rate semantics are
  preserved with no graph changes.
- `<break>` replays upstream's in-graph `dur[sid] = sdur` override via
  a two-pass `durs_rate` rewrite: pass 2 rewrites only the break
  token's rate so `round(base/rate)` hits the requested frame count,
  and the graph's own sos/tail caps then apply exactly as they do after
  upstream's override (verified frame-exact against the scripted
  forward). The mel pause region is filled by the numpy `fx_pauses`
  port with this set's `sil_value` (`-5.0`, declared by
  `ModelSpec.pause_sil_value` / `Version_V56.pause_sil_value`, dumped
  from the checkpoint; v5_ru uses `-8.0`).
- Comma breaks take the same override+silence path: upstream's batch-2
  support re-prediction only runs when `gt_durs` is absent, which the
  always-gt operating point never is (measured in the scripted
  forward).
- `<prosody pitch>` fills the per-token `pitch_coefs` input; the
  acoustic graph applies the checkpoint's `update_pitch_coef`
  (scale plus speaker mean/std shift, exact zeros stay silent). The
  L0/L1/L2 and SSML e2e gates (`high`, `low`, `robot`) validate the
  transform against torch.
- `<intensity=N>` is stripped before XML parsing (upstream's
  `extract_intensity_from_ssml` order) and scales the `*focus*` mask;
  the first tag wins for the whole utterance (differing values warn);
  with no `*`/`**` marks it is inert (upstream audio is identical) and
  warns loudly.
