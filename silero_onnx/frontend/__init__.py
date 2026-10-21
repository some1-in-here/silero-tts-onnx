"""Text frontend package (language injected, version-independent).

Legacy names from the former ``frontend.py`` module are re-exported so
existing callers keep working unchanged.
"""
from ..languages.base import add_sos_eos, collapse_whitespace, tokenize
from ..languages.ru.abbreviations import ABBR, expand_abbreviations
from ..languages.ru.numbers import expand_numbers
from ..languages.ru.stress import apply_stress, get_accentor
from ..languages.ru.symbols import (DATA_DIR, load_speaker_map,
                                    load_symbol_table)
from .base import AbstractTextFrontend
from .pipeline import TextPipeline
from .registry import FrontendRegistry
from .ru import TextFrontend, TextFrontend_Ru, TorchFreeFrontend
from .stages import (AbbrevStage, NormalizeStage, NumberStage, StressStage,
                     TokenizeStage)

FrontendRegistry.register(TextFrontend_Ru)


def normalize(text: str) -> str:
    """Deterministic text cleanup: whitespace, abbreviations, numbers."""
    text = collapse_whitespace(text)
    return expand_numbers(expand_abbreviations(text))


def prepare(text: str, symbol_to_id, stress=True):
    """Full prototype chain: normalize -> stress -> token ids."""
    t = normalize(text)
    if stress:
        t = apply_stress(t)
    return tokenize(t, symbol_to_id)


__all__ = [
    "DATA_DIR", "AbstractTextFrontend", "TextFrontend", "TextFrontend_Ru",
    "TorchFreeFrontend", "TextPipeline", "FrontendRegistry", "NormalizeStage",
    "AbbrevStage", "NumberStage", "StressStage", "TokenizeStage", "normalize",
    "prepare", "expand_abbreviations", "expand_numbers", "apply_stress",
    "get_accentor", "load_symbol_table", "load_speaker_map", "tokenize",
    "add_sos_eos", "ABBR",
]
