"""Strategy registry/factory + backward-compatible public facade hooks.

``StrategyFactory`` is the only place that maps a version id to a
concrete strategy class. The engine creates the language/frontend/
sessions and passes them in (dependency injection).
"""
from ..versions import DEFAULT_VERSION
from .base import AbstractTTSStrategy, TTSStrategy
from .v5 import Strategy_V5
from .v5_6 import Strategy_V56, Strategy_V5_6

__all__ = ["AbstractTTSStrategy", "TTSStrategy", "Strategy_V5",
           "Strategy_V56", "Strategy_V5_6", "StrategyFactory",
           "create_strategy", "SileroTTS"]


class StrategyFactory:
    """Version-id -> strategy class registry."""

    _strategies = {
        "v5_ru": Strategy_V5,
        "v5_6_ru": Strategy_V56,
    }

    @classmethod
    def register(cls, version_id, strategy_cls):
        if version_id in cls._strategies:
            raise ValueError(f"strategy '{version_id}' already registered")
        cls._strategies[version_id] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, version_id):
        if version_id not in cls._strategies:
            raise ValueError(
                f"no strategy for version '{version_id}', available: "
                f"{sorted(cls._strategies)}")
        return cls._strategies[version_id]

    @classmethod
    def create(cls, version_id, *args, **kwargs):
        return cls.get(version_id)(*args, **kwargs)

    @classmethod
    def list(cls):
        return sorted(cls._strategies)


#: Historical private registry name used by tests/tools.
_STRATEGIES = StrategyFactory._strategies


def create_strategy(version=DEFAULT_VERSION, model_dir=None, data_dir=None):
    """Create a strategy through the engine factory (compat helper)."""
    from ..engine import EngineFactory
    return EngineFactory.create_engine(
        version_id=version, model_dir=model_dir,
        data_dir=data_dir).strategy


def __getattr__(name):
    # Lazy import avoids an engine <-> strategies import cycle.
    if name == "SileroTTS":
        from ..engine import SileroTTS
        return SileroTTS
    raise AttributeError(name)
