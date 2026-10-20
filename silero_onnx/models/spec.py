"""Declarative model-version specification (single source of truth).

``ModelSet`` describes the ONNX artifacts and I/O topology.
``ModelSpec`` adds capability flags used by EngineFactory/strategies.
Nothing here imports a language or a strategy.
"""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class ModelSet:
    """The files and graph topology of one model generation."""

    model_files: Dict[str, str]
    session_topology: Tuple[str, ...]
    acoustic_inputs: Tuple[str, ...]
    shared_models: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"stress", "ngram"}))


@dataclass(frozen=True)
class ModelSpec:
    """A version's model set plus its declared capabilities."""

    version_id: str
    display_name: str
    model_set: ModelSet
    #: True when the acoustic graph accepts stress/focus features.
    supports_stress: bool = True
    #: True when the runtime may parse SSML for this version.
    supports_ssml: bool = True
    #: True when <prosody pitch> has a real channel (pitch_coefs).
    supports_pitch: bool = False
    #: True when the version supports question-type sentence classification.
    supports_questions: bool = False
    #: Acoustic graph takes a pitch_coefs input. v5 uses it directly;
    #: v5_6 applies the checkpoint's update_pitch_coef to raw gt_pitch.
    has_pitch_coefs_input: bool = False
    requires_gt_durs: bool = False
    requires_gt_pitch: bool = False
    #: Mel silence fill dumped from the upstream checkpoint (fx_pauses).
    pause_sil_value: float = -8.0

    @property
    def model_files(self) -> Dict[str, str]:
        return self.model_set.model_files

    @property
    def session_topology(self) -> Tuple[str, ...]:
        return self.model_set.session_topology

    @property
    def acoustic_inputs(self) -> Tuple[str, ...]:
        return self.model_set.acoustic_inputs

    @property
    def shared_models(self) -> FrozenSet[str]:
        return self.model_set.shared_models

    def get_model_path(self, model_dir: str, key: str) -> str:
        import os
        if key not in self.model_files:
            raise KeyError(f"model '{key}' not in spec for {self.version_id}")
        return os.path.join(model_dir, self.model_files[key])

    def validate_model_dir(self, model_dir: str) -> List[str]:
        import os
        return [fname for fname in self.model_files.values()
                if not os.path.isfile(os.path.join(model_dir, fname))]


# Future versions (not registered).
# _EMPTY_SET = ModelSet(model_files={}, session_topology=(),
#                       acoustic_inputs=())

_MODEL_SET_V5 = ModelSet(
    model_files={
        "acoustic": "acoustic_v5_ru.onnx",
        "vocoder": "vocoder_v5_ru.onnx",
        "istft": "istft_48k.onnx",
        "stress": "silero_stress.onnx",
        "ngram": "ngram_core.onnx",
    },
    session_topology=("stress", "ngram", "acoustic", "vocoder", "istft"),
    acoustic_inputs=("sequence", "speaker_ids", "durs_rate", "pitch_coefs"),
)

_MODEL_SET_V56 = ModelSet(
    model_files={
        "acoustic": "acoustic_v5_6_ru.onnx",
        "dur_predictor": "dur_predictor_v5_6_ru.onnx",
        "pitch_predictor": "pitch_predictor_v5_6_ru.onnx",
        "vocoder": "vocoder_v5_6_ru.onnx",
        "stress": "silero_stress.onnx",
        "ngram": "ngram_core.onnx",
    },
    session_topology=("stress", "ngram", "dur_predictor", "pitch_predictor",
                      "acoustic", "vocoder"),
    acoustic_inputs=("sequence", "speaker_ids", "durs_rate", "gt_durs",
                     "gt_pitch", "pitch_coefs", "type_ids", "focus_mask"),
)

MODEL_SPECS: Dict[str, ModelSpec] = {
    "v5_ru": ModelSpec(
        version_id="v5_ru",
        display_name="Silero TTS v5 Russian",
        model_set=_MODEL_SET_V5,
        supports_stress=True,
        supports_ssml=True,
        supports_pitch=True,
        supports_questions=False,
        has_pitch_coefs_input=True,
        requires_gt_durs=False,
        requires_gt_pitch=False,
        pause_sil_value=-8.0,
    ),
    "v5_6_ru": ModelSpec(
        version_id="v5_6_ru",
        display_name="Silero TTS v5.6 Russian (emphasis)",
        model_set=_MODEL_SET_V56,
        supports_stress=True,
        supports_ssml=True,
        supports_pitch=True,
        supports_questions=True,
        has_pitch_coefs_input=True,
        requires_gt_durs=True,
        requires_gt_pitch=True,
        pause_sil_value=-5.0,
    ),
    # Future versions (not registered):
    # "v4_ru": ModelSpec(
    #     version_id="v4_ru",
    #     display_name="Silero TTS v4 Russian",
    #     model_set=_EMPTY_SET,
    #     supports_stress=False,
    #     supports_ssml=False,
    # ),
    # "v3_ru": ModelSpec(
    #     version_id="v3_ru",
    #     display_name="Silero TTS v3 Russian",
    #     model_set=_EMPTY_SET,
    #     supports_stress=False,
    #     supports_ssml=False,
    # ),
}


def get_spec(version_id: str) -> ModelSpec:
    if version_id not in MODEL_SPECS:
        raise ValueError(f"unknown model version '{version_id}', available: "
                         f"{list_versions()}")
    return MODEL_SPECS[version_id]


def list_versions() -> List[str]:
    return sorted(MODEL_SPECS)


__all__ = ["ModelSpec", "ModelSet", "MODEL_SPECS", "get_spec",
           "list_versions"]
