"""Language registry: add a language without touching existing code."""
from typing import Callable, Dict, List

from .base import AbstractLanguage


class LanguageRegistry:
    """Registry of language factories keyed by ``language_id``.

    Factories are classes (or callables) accepting optional
    ``models_dir`` / ``data_dir`` keyword arguments, so the engine can
    construct a language instance on demand.
    """

    _factories: Dict[str, Callable[..., AbstractLanguage]] = {}

    @classmethod
    def register(cls, language_id: str = None, factory: Callable = None):
        """Register a language class/factory.

        Usable directly (``register("xx", Language_Xx)``) or as a
        decorator (``@LanguageRegistry.register()`` on a class with a
        ``language_id`` attribute).
        """
        if factory is None and isinstance(language_id, type) and \
                issubclass(language_id, AbstractLanguage):
            factory = language_id
            language_id = factory.language_id
        if factory is None:
            def _decorator(cls_):
                cls.register(cls_.language_id, cls_)
                return cls_
            return _decorator
        if not language_id:
            language_id = getattr(factory, "language_id", "")
        if not language_id:
            raise ValueError("language_id is required")
        if language_id in cls._factories:
            raise ValueError(f"language '{language_id}' already registered")
        cls._factories[language_id] = factory
        return factory

    @classmethod
    def get(cls, language_id: str, **kwargs) -> AbstractLanguage:
        if language_id not in cls._factories:
            raise ValueError(
                f"unknown language '{language_id}', available: "
                f"{sorted(cls._factories)}")
        return cls._factories[language_id](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._factories)

    @classmethod
    def clear(cls) -> None:
        cls._factories.clear()

    @classmethod
    def unregister(cls, language_id: str) -> None:
        cls._factories.pop(language_id, None)


__all__ = ["LanguageRegistry"]
