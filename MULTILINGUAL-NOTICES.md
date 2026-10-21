# Multilingual contribution notices

This contribution adapts Vladimir Govorov / some1-in-here's ONNX work
(starting at c62b8f951a346e3db2d0d0a03e3e581143bf7d52), with model
architecture and weights from the Silero Team's `snakers4/silero-models`.
Jim Grimsby Jr. performed the multilingual adaptation and Windows/NVDA
evaluation with extensive AI assistance. No claim of authorship of the
original models is made.

The contributed port/runtime code follows this repository's
CC BY-NC-SA 4.0 LICENSE. Existing upstream notices and licenses remain.
Do not apply this code license as a blanket replacement for model terms or
third-party dependency licenses.

## Model terms

The official Silero README says its models follow its main license except
the base CIS models, which are MIT. The inventory records official download
URLs, source SHA-256 identities and license links for the exact checkpoints.

- `v5_cis_base_nostress` weights: MIT, retain `LICENSE_CIS` from Silero.
- The other nine offered checkpoints: Silero's main CC BY-NC-SA 4.0 license.
- The separate runtime/ports remain CC BY-NC-SA 4.0 even when used with MIT
  weights. This does not make those weights noncommercial by themselves.

Sources checked on 2026-10-18:

- https://github.com/snakers4/silero-models/blob/master/README.md#licence
- https://github.com/snakers4/silero-models/blob/master/LICENSE
- https://github.com/snakers4/silero-models/blob/master/LICENSE_CIS

An ONNX conversion changes representation, not permission. Retain original
attribution, license text, checkpoint identity and a description of changes
alongside each downloadable model. No new commercial-use permission is
asserted here.

## Optional preprocessing and recordings

No Aksharamukha, Sonata/NVDA GPL code, private logs, user recordings,
embeddings or personal voice banks are bundled in this contribution.
Indic native-script evaluation used Aksharamukha 2.3 separately; it is
AGPL-3.0 and is not relicensed by this repository. Generated listening
recordings use fixed published Silero voices and public/example or authored
test sentences, not private cloned voices. They are evaluation material;
they do not establish native-speaker approval.
