"""Run the five v5_ru model exporters in dependency order.

Order is fixed: acoustic -> vocoder -> istft -> stress -> ngram. Stops
at the first failing step (non-zero exit propagates). Each step writes
its own *_result.txt log next to the models.

``ExportPipeline_V5`` is the concrete Template Method participant; the
shared prepare/validate/session checks live in ``scripts/strategies/base.py``.

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export.py [--out-dir onnx]
"""
import argparse
import os
import subprocess
import sys

V5_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.dirname(V5_DIR)
SCRIPTS_DIR = os.path.dirname(STRATEGIES_DIR)
ROOT = os.path.dirname(SCRIPTS_DIR)
for _p in (V5_DIR, STRATEGIES_DIR, SCRIPTS_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import AbstractExportPipeline  # noqa: E402

STEPS = ["export_acoustic.py", "export_vocoder.py", "export_istft.py",
         "export_stress.py", "export_ngram.py"]


class ExportPipeline_V5(AbstractExportPipeline):
    """Export the complete v5_ru model set (five single-file ONNX models)."""

    version_id = "v5_ru"
    artifacts = ("acoustic", "vocoder", "istft", "stress", "ngram")

    def __init__(self, out_dir):
        super().__init__(out_dir, self.artifacts)

    def export(self):
        for step in STEPS:
            print(f"=== v5/{step} ===", flush=True)
            r = subprocess.run([sys.executable, os.path.join(V5_DIR, step),
                                "--out-dir", self.out_dir])
            if r.returncode != 0:
                print(f"export(v5): {step} FAILED (exit {r.returncode})",
                      flush=True)
                raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="export all five v5_ru models")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(V5_DIR))),
        "models", "onnx"))
    a = ap.parse_args()
    ExportPipeline_V5(a.out_dir).run()


if __name__ == "__main__":
    main()
