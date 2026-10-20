"""Language layer: DI-friendly, version-independent text rules."""
from .base import AbstractLanguage, add_sos_eos, collapse_whitespace, tokenize
from .registry import LanguageRegistry
from .ru import Language_Ru

LanguageRegistry.register("ru", Language_Ru)

__all__ = ["AbstractLanguage", "Language_Ru", "LanguageRegistry",
           "add_sos_eos", "collapse_whitespace", "tokenize"]
