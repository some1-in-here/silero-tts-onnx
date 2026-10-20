"""Abstract ONNX export pipeline (Template Method).

Concrete pipelines implement only ``export``; the shared algorithm is:

    prepare()   create the destination (build output, never versioned)
    export()    run the version-specific save/export steps
    validate()  check every expected artifact and open an ORT session
    finalize()  print the summary

Paths and capabilities come from ``silero_onnx.versions.MODEL_SETS``;
no exporter hardcodes model filenames. Build-time only: this module may
import torch-free runtime metadata, but never imports torch itself.
"""
import os
import sys
from abc import ABC, abstractmethod

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(SCRIPTS_DIR)
for _p in (THIS_DIR, SCRIPTS_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from silero_onnx.versions import MODEL_SETS  # noqa: E402  (torch-free)


class AbstractExportPipeline(ABC):
    """Version-independent export algorithm with version-specific steps."""

    version_id = ""
    #: Open every produced file with onnxruntime as a final structural gate.
    validate_sessions = True

    def __init__(self, out_dir, artifacts):
        self.out_dir = os.path.abspath(out_dir)
        self.model_files = dict(MODEL_SETS.get(self.version_id, {}))
        self.artifact_keys = list(artifacts)
        missing = [k for k in self.artifact_keys if k not in self.model_files]
        if missing:
            raise ValueError(
                f"export({self.version_id}): unknown artifact keys {missing}; "
                f"known: {sorted(self.model_files)}")

    @property
    def artifact_paths(self):
        return [os.path.join(self.out_dir, self.model_files[k])
                for k in self.artifact_keys]

    def prepare(self):
        """Create the build output directory (ignored by git)."""
        os.makedirs(self.out_dir, exist_ok=True)

    @abstractmethod
    def export(self):
        """Version-specific export/save steps (implemented by subclasses)."""

    def validate(self):
        """Common postcondition: files exist and every ONNX session opens."""
        missing = [p for p in self.artifact_paths if not os.path.isfile(p)]
        if missing:
            raise SystemExit(
                f"export({self.version_id}): missing artifacts: "
                + ", ".join(os.path.basename(p) for p in missing))
        if not self.validate_sessions:
            return
        import onnxruntime as ort
        for path in self.artifact_paths:
            try:
                ort.InferenceSession(path,
                                     providers=["CPUExecutionProvider"])
            except Exception as e:
                raise SystemExit(
                    f"export({self.version_id}): cannot open "
                    f"{os.path.basename(path)} as an ONNX session: {e}")

    def finalize(self):
        print(f"export({self.version_id}): {len(self.artifact_paths)} models "
              f"written to {self.out_dir}", flush=True)

    def run(self):
        """Fixed order: prepare -> export -> validate -> finalize."""
        self.prepare()
        self.export()
        self.validate()
        self.finalize()
        return self.artifact_paths


__all__ = ["AbstractExportPipeline"]
