"""Model-version layer: declarative specs + registry."""
from .registry import ModelRegistry
from .spec import (MODEL_SPECS, ModelSet, ModelSpec, get_spec,
                   list_versions)
from .versions import AbstractModelVersion, Version_V5Ru, Version_V56
# Future exports: from .versions import Version_V3, Version_V4

__all__ = [
    "ModelSpec", "ModelSet", "MODEL_SPECS", "get_spec", "list_versions",
    "AbstractModelVersion", "Version_V5Ru", "Version_V56", "ModelRegistry",
    # Future exports: "Version_V3", "Version_V4",
]
