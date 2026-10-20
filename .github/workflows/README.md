# GitHub Actions workflows

Three pipelines, split by cost. Architecture rationale: `DEVELOPMENT.md` §5.

| Workflow | Trigger | What it runs |
|---|---|---|
| [`ci.yml`](ci.yml) | every push (any branch) + pull requests (+ manual) | torch-free, model-free: install runtime + dev extras, torch-free import and light CLI smoke checks, `pytest` — model-dependent tests skip and are reported in one summary line (per-test reasons available with `-rs`; see [`../../tests/conftest.py`](../../tests/conftest.py)) |
| [`e2e.yml`](e2e.yml) | every push to `master` (any merge included) + manual (`workflow_dispatch`, any branch) | full pipeline: build env (CPU torch) → `export_all.py` (both model sets) → `check_models.py` (all 9 artifacts) → `check_parity.py` (v5 e2e; the v5_6 e2e gate runs inside its export step) → full `pytest --require-models` → `package_models.py` dry-run → gate-log summary + upload |
| [`export_and_release.yml`](export_and_release.yml) | `v*` tags (or manual with a `version` input) | the same export/validate/test chain, then packages the release ZIP + individual models and publishes the GitHub Release; any gate failure stops the chain before publishing |

## Running the full E2E

Automatic: every push to `master` (the default branch) runs the full
pipeline — merges included.

Manual on any branch (no tag or merge needed): **Actions → Full E2E
(export + validate + test) → Run workflow**, then pick the branch in the
**Use workflow from** dropdown. The run is read-only: it builds and
validates, it never publishes anything.

## Notes

- **Models are data, not code.** `models/` is git-ignored; the required
  artifacts are listed in `silero_onnx.versions.MODEL_SETS` (5 `v5_ru` +
  6 `v5_6_ru` entries; 9 unique files, with stress/ngram shared), and
  `scripts/package_models.py` derives the release ZIP from that list — a
  release cannot silently omit a model set.
- **Test split:** 92/131 tests run without models everywhere; the other 39
  need a built `models/onnx/` (or `SILERO_ONNX_MODELS`) and carry the
  `requires_models` / `requires_v56` markers. `e2e.yml` and the release
  always have the models and pass `--require-models`, so a missing set
  fails the job there instead of silently skipping.
- **Failures are explicit:** every step fails the job on error; `e2e.yml`
  and the release summarize the per-step gate logs (`*_result.txt`),
  annotate failing gates, and attach the logs as artifacts.
