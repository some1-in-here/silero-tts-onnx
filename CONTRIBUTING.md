<!-- Audience: developers and AI agents; the contract for documentation
     layering (what belongs in README.md vs DEVELOPMENT.md vs scripts docs). -->

# Contributing

Project-wide rules for changes. Where everything lives:

- Runtime internals, architecture, measurements, CI rationale —
  [`DEVELOPMENT.md`](DEVELOPMENT.md)
- Build/export walkthrough and per-script reference —
  [`scripts/README.md`](scripts/README.md) and
  [`scripts/DEVELOPMENT.md`](scripts/DEVELOPMENT.md)
- CI pipelines and how to run them —
  [`.github/workflows/README.md`](.github/workflows/README.md)

## Documentation layers (keep them separate)

| File | Audience | Belongs there |
|---|---|---|
| `README.md` | users | what it is, install, model download, usage examples, env vars, license |
| `DEVELOPMENT.md` | contributors | architecture, how-it-works, internals, measurements, CI |
| `scripts/README.md` | builders | from-zero export walkthrough, per-script table |
| `scripts/DEVELOPMENT.md` | export maintainers | exporter internals, gates, thresholds |
| `.github/workflows/README.md` | CI maintainers | pipeline reference, run instructions |

**`README.md` rules (hard):**

- **Strictly user-facing.** A reader who just wants speech out of text must
  never have to parse internals.
- No "how it works" paragraphs, architecture, model-file/session
  descriptions, numeric measurements or gate results, CI/workflow details.
- Keep blocks short and skimmable: brief paragraphs, tables for options,
  copy-paste snippets, examples over explanation.
- Keep the established section order: Quickstart (copy-paste) → Features →
  Install → Models → Usage → Emphasis / SSML → Environment variables →
  Tests → License.
- Tempted to add a technical detail? Put it in `DEVELOPMENT.md` (or the
  matching `scripts/*` doc) and, at most, link to it from the README.

## Code and tests

- The runtime (`silero_onnx/`) and `tests/` stay **100% torch-free** — a
  tested contract, not a preference (torch is allowed only under
  `scripts/`, in a separate build env).
- Run `pytest tests/ -v` before opening a PR. Model-dependent tests need
  `models/onnx/` (or `SILERO_ONNX_MODELS`); without it they skip with an
  explicit reason.
- CI: the model-free suite runs on every push/PR; the full E2E (both model
  sets) runs on every `master` push and on demand — see the workflows
  README.
- Never commit models, gate logs or generated fixtures (`models/` and
  runtime caches are git-ignored).
- Behavior changes come with a numeric gate; a green suite is the
  definition of done ([`DEVELOPMENT.md`](DEVELOPMENT.md) §6).
