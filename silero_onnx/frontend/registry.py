"""Frontend registry: add a language frontend without editing the factory."""
from typing import Callable, Dict, List, Optional

from .base import AbstractTextFrontend


class FrontendRegistry:
    """Registry of frontend factories keyed by ``frontend_id``.

    A language-specific frontend registers itself under the language id
    (``"ru" -> TextFrontend_Ru``). ``EngineFactory`` resolves it after the
    language and version, so adding a language means adding a registration
    entry, not an ``if language_id == ...`` branch. Unknown languages fall
    back to the generic :class:`~silero_onnx.frontend.ru.TextFrontend` when
    the caller supplies it as ``default``.
    """

    _factories: Dict[str, Callable[..., AbstractTextFrontend]] = {}

    @classmethod
    def register(cls, frontend_id: Optional[str] = None,
                 factory: Optional[Callable] = None):
        """Register a frontend class/factory (directly or as a decorator)."""
        if factory is None and isinstance(frontend_id, type) and \
                issubclass(frontend_id, AbstractTextFrontend):
            factory = frontend_id
            frontend_id = factory.frontend_id
        if factory is None:
            def _decorator(cls_):
                cls.register(cls_.frontend_id, cls_)
                return cls_
            return _decorator
        if not frontend_id:
            frontend_id = getattr(factory, "frontend_id", "")
        if not frontend_id:
            raise ValueError("frontend_id is required")
        if frontend_id in cls._factories:
            raise ValueError(f"frontend '{frontend_id}' already registered")
        cls._factories[frontend_id] = factory
        return factory

    @classmethod
    def get(cls, frontend_id: str, default=None, **kwargs):
        """Instantiate the frontend for ``frontend_id`` (or ``default``)."""
        factory = cls._factories.get(frontend_id)
        if factory is None:
            if default is None:
                raise ValueError(
                    f"unknown frontend '{frontend_id}', available: "
                    f"{sorted(cls._factories)}")
            factory = default
        return factory(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._factories)

    @classmethod
    def clear(cls) -> None:
        cls._factories.clear()

    @classmethod
    def unregister(cls, frontend_id: str) -> None:
        cls._factories.pop(frontend_id, None)


__all__ = ["FrontendRegistry"]
