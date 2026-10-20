"""Concrete model versions (``Version_<Id>`` naming contract).

A version knows its ONNX files, session topology, acoustic inputs and
declared capabilities. It never imports a language, frontend or strategy.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from .registry import ModelRegistry
from .spec import MODEL_SPECS, ModelSpec


class AbstractModelVersion(ABC):
    """Interface shared by every model version."""

    version_id: str = ""

    @property
    @abstractmethod
    def spec(self) -> ModelSpec:
        ...

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def supports_stress(self) -> bool:
        return self.spec.supports_stress

    @property
    def supports_ssml(self) -> bool:
        return self.spec.supports_ssml

    @property
    def supports_pitch(self) -> bool:
        return self.spec.supports_pitch

    @property
    def supports_questions(self) -> bool:
        return self.spec.supports_questions

    @property
    def pause_sil_value(self) -> float:
        return self.spec.pause_sil_value

    def has_pitch_coefs_input(self) -> bool:
        return self.spec.has_pitch_coefs_input

    def requires_gt_durs(self) -> bool:
        return self.spec.requires_gt_durs

    @property
    def requires_gt_pitch(self) -> bool:
        return self.spec.requires_gt_pitch

    def model_files(self) -> Dict[str, str]:
        return dict(self.spec.model_files)

    def session_topology(self) -> List[str]:
        return list(self.spec.session_topology)

    def acoustic_inputs(self) -> List[str]:
        return list(self.spec.acoustic_inputs)

    def validate_model_dir(self, model_dir: str) -> List[str]:
        return self.spec.validate_model_dir(model_dir)

    def get_model_path(self, model_dir: str, key: str) -> str:
        return self.spec.get_model_path(model_dir, key)


class Version_V5Ru(AbstractModelVersion):
    """Silero TTS v5 Russian (classic pipeline)."""

    version_id = "v5_ru"
    _spec = MODEL_SPECS["v5_ru"]

    @property
    def spec(self) -> ModelSpec:
        return self._spec


class Version_V56(AbstractModelVersion):
    """Silero TTS v5.6 Russian (emphasis-aware pipeline)."""

    version_id = "v5_6_ru"
    _spec = MODEL_SPECS["v5_6_ru"]

    @property
    def spec(self) -> ModelSpec:
        return self._spec


# Future versions (not registered).
# class Version_V4(AbstractModelVersion):
#     """Silero TTS v4 Russian (classic pipeline)."""
#
#     version_id = "v4_ru"
#     _spec = MODEL_SPECS["v4_ru"]
#
#     @property
#     def spec(self) -> ModelSpec:
#         return self._spec
#
#
# class Version_V3(AbstractModelVersion):
#     """Silero TTS v3 Russian (classic pipeline)."""
#
#     version_id = "v3_ru"
#     _spec = MODEL_SPECS["v3_ru"]
#
#     @property
#     def spec(self) -> ModelSpec:
#         return self._spec


# Register built-ins once. Adding a version = one class + one register call.
for _version in (Version_V5Ru(), Version_V56()):
    ModelRegistry.register(_version)
del _version


__all__ = ["AbstractModelVersion", "Version_V5Ru", "Version_V56"]
# Future exports: "Version_V4", "Version_V3"
