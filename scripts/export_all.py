"""Run fixture generation + version exporters in dependency order.

Order is fixed: make_fixtures -> v5 (5 models) -> v5_6 (4 models).
Stops at the first failing step (non-zero exit propagates). Each step
writes its own *_result.txt log next to the models.

Usage:
  python scripts/export_all.py [--out-dir onnx]
"""
import argparse
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = ["make_fixtures.py", os.path.join("strategies", "v5", "export.py"),
         os.path.join("strategies", "v5_6", "export.py")]


def main():
    ap = argparse.ArgumentParser(description="export all ONNX models "
                                             "(v5_ru x5 + v5_6_ru x4)")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(SCRIPTS_DIR), "models", "onnx"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for step in STEPS:
        print(f"=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, step),
                            "--out-dir", a.out_dir])
        if r.returncode != 0:
            print(f"export_all: {step} FAILED (exit {r.returncode})",
                  flush=True)
            raise SystemExit(r.returncode)
    print(f"export_all: all 9 models written to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
