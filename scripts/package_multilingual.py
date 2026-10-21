"""Package reviewed model data only; never uploads or publishes anything."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data = json.loads((args.model_dir / "frontend.json").read_text(encoding="utf-8"))
    inventory = json.loads((root / "scripts/multilingual-checkpoints.json").read_text(encoding="utf-8"))
    identity = inventory[data["id"]]
    if data["source_sha256"] != identity["sha256"]:
        raise ValueError("Source checkpoint identity is not reviewed")
    names = {"acoustic.onnx", "vocoder.onnx"}
    if data["spectral"]:
        names.add("istft.onnx")
    if set(data["graphs"]) != names:
        raise ValueError("Graph manifest is incomplete or contains unexpected files")
    for name in names:
        path = args.model_dir / name
        record = data["graphs"][name]
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Graph hash/size mismatch: " + name)
    # Strict allowlist: no Python, executable checkpoint, logs or personal data.
    notice = (root / "MULTILINGUAL-NOTICES.md").read_text(encoding="utf-8")
    license_file = root / ("licenses/silero-LICENSE_CIS" if identity["license"] == "MIT" else "LICENSE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(names):
            z.write(args.model_dir / name, name)
        z.write(args.model_dir / "frontend.json", "frontend.json")
        z.write(license_file, "MODEL-LICENSE.txt")
        z.writestr("ATTRIBUTION.md", notice)
        z.writestr("SOURCE.json", json.dumps(identity, indent=2))
    with zipfile.ZipFile(args.output) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    print(json.dumps(dict(file=args.output.name, sha256=digest(args.output),
                          bytes=args.output.stat().st_size), indent=2))


if __name__ == "__main__":
    main()
