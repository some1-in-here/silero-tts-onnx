"""Registry of model versions (open for extension, closed for editing)."""
from typing import Dict, List


class ModelRegistry:
    """Version-id -> AbstractModelVersion instance registry."""

    _versions: Dict[str, object] = {}

    @classmethod
    def register(cls, version) -> None:
        vid = version.version_id
        if vid in cls._versions:
            raise ValueError(f"version '{vid}' already registered")
        cls._versions[vid] = version

    @classmethod
    def unregister(cls, version_id: str) -> None:
        cls._versions.pop(version_id, None)

    @classmethod
    def get(cls, version_id: str):
        if version_id not in cls._versions:
            raise ValueError(
                f"unknown model version '{version_id}', available: "
                f"{sorted(cls._versions)}")
        return cls._versions[version_id]

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._versions)

    @classmethod
    def clear(cls) -> None:
        cls._versions.clear()


__all__ = ["ModelRegistry"]
