"""CLI for additive multilingual exports; does not alter Russian strategies."""
import runpy
import sys
from pathlib import Path


def main():
    if not __debug__:
        raise RuntimeError("Export validation requires assertions; do not use Python -O")
    # torch.onnx prints Unicode status marks; Windows redirected output may
    # otherwise default to a legacy encoding. Do not change library imports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parent.parent
    strategy = root / "scripts" / "strategies" / "multilingual"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(strategy))
    runpy.run_path(str(strategy / "export.py"), run_name="__main__")


if __name__ == "__main__":
    main()
