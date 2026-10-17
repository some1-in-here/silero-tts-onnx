"""Pack a built model directory into the release ZIP (torch-free).

The expected file list comes EXCLUSIVELY from
silero_onnx.versions.MODEL_SETS (every artifact every supported version
needs), so a release can never silently omit a model set. Missing files
abort the pack BEFORE anything is written; the zip is flat (basenames
only), one entry per required model.

Used by the release workflow and the manual full-E2E workflow (packaging
dry-run): both build into models/onnx/ first, then package.

Usage:
  python scripts/package_models.py [--models models/onnx]
                                   [--zip models/silero_onnx_models.zip]
"""
import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def expected_names():
    """Every .onnx filename any supported version needs (sorted)."""
    from silero_onnx.versions import MODEL_SETS  # torch-free
    return sorted({f for files in MODEL_SETS.values() for f in files.values()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "onnx"),
                    help="built model directory (from export_all.py)")
    ap.add_argument("--zip", dest="zip_path",
                    default=os.path.join(ROOT, "models",
                                         "silero_onnx_models.zip"),
                    help="output zip path (inside git-ignored models/ "
                         "by default)")
    a = ap.parse_args()

    names = expected_names()
    missing = [n for n in names
               if not os.path.isfile(os.path.join(a.models, n))]
    if missing:
        raise SystemExit(
            "package_models: FAILED: missing " + ", ".join(missing)
            + " in " + a.models + " (MODEL_SETS lists them as required; "
            "run scripts/export_all.py first)")
    with zipfile.ZipFile(a.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(a.models, n), n)
    print("package_models: %d models packed -> %s (%.1f MB)"
          % (len(names), a.zip_path, os.path.getsize(a.zip_path) / 1e6),
          flush=True)
    for n in names:
        print("  " + n, flush=True)


if __name__ == "__main__":
    main()
