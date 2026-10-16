"""Run the five v5_ru model exporters in dependency order.

Order is fixed: acoustic -> vocoder -> istft -> stress -> ngram. Stops
at the first failing step (non-zero exit propagates). Each step writes
its own *_result.txt log next to the models.

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export.py [--out-dir onnx]
"""
import argparse
import os
import subprocess
import sys

V5_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = ["export_acoustic.py", "export_vocoder.py", "export_istft.py",
         "export_stress.py", "export_ngram.py"]


def main():
    ap = argparse.ArgumentParser(description="export all five v5_ru models")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(V5_DIR))), "models", "onnx"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for step in STEPS:
        print(f"=== v5/{step} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(V5_DIR, step),
                            "--out-dir", a.out_dir])
        if r.returncode != 0:
            print(f"export(v5): {step} FAILED (exit {r.returncode})",
                  flush=True)
            raise SystemExit(r.returncode)
    print(f"export(v5): all 5 models written to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
