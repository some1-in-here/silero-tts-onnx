"""Torch-free check of graph hashes and paired export fixtures.

Use --model-dir and --fixtures-dir from export_multilingual.py. This is a
numeric gate, not a native pronunciation or screen-reader acceptance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from silero_onnx.multilingual import MultilingualRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--lower-memory", action="store_true")
    args = parser.parse_args()
    data = json.loads((args.model_dir / "frontend.json").read_text(encoding="utf-8"))
    for name, identity in data["graphs"].items():
        if name not in {"acoustic.onnx", "vocoder.onnx", "istft.onnx"}:
            raise ValueError("Unexpected graph name in manifest")
        path = args.model_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError("Graph identity mismatch: " + name)
    cases = json.loads((args.fixtures_dir / "export-parity.json").read_text(encoding="utf-8"))["cases"]
    if not cases:
        raise ValueError("No parity cases: refusing a vacuous pass")
    tts = MultilingualRuntime(args.model_dir, args.lower_memory)
    results = []
    for index, case in enumerate(cases):
        audio, _ = tts.synthesize(case["text"], case["speaker"], case.get("lang"))
        reference, rate = sf.read(args.fixtures_dir / f"reference-{index}.wav", dtype="float32")
        if rate != 48000 or reference.shape != audio.shape:
            raise ValueError("Reference waveform format/length mismatch")
        error = float(np.max(np.abs(audio - reference)))
        if error >= 0.005:
            raise ValueError(f"Reference waveform difference exceeds gate: {error}")
        results.append(dict(speaker=case["speaker"], lang=case.get("lang"),
                            samples=len(audio), max_wave_difference=error))
    print(json.dumps(dict(passed=True, lower_memory=args.lower_memory, cases=results), indent=2))


if __name__ == "__main__":
    main()
