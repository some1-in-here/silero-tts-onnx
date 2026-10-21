# Multilingual contribution work

- Preserve the existing Russian strategies and source baseline. Multilingual
  fixed-voice support is additive and does not claim complete SSML/stress parity.
- Runtime and tests stay torch-free. PyTorch is export-time only.
- Model binaries, executable checkpoints, generated audio, logs and private
  NVDA configurations never enter Git. Release artifacts need explicit approval.
- Retain component-specific model licenses and attribution, hashes and sample
  provenance. Do not imply conversion parity is native-language approval.
- Serialize inference; two CPU threads and one inter-op for reproducibility.
- Public evidence must be path-redacted and distinguish retained observations
  from current-branch checks. No user voice banks or raw private logs.
- Publication needs explicit repository-owner approval for the exact artifacts
  and destination. Local validation and preservation commits are separate.
