"""Composable text-pipeline stages.

The stages are deliberately small and language-agnostic; all language
knowledge comes from the injected ``AbstractLanguage``.
"""
from ..languages.base import add_sos_eos, collapse_whitespace, tokenize


class NormalizeStage:
    """Collapse whitespace (the deterministic first step)."""

    name = "normalize"

    def __call__(self, text):
        return collapse_whitespace(text)


class AbbrevStage:
    name = "abbreviations"

    def __init__(self, language):
        self._language = language

    def __call__(self, text):
        return self._language.expand_abbreviations(text)


class NumberStage:
    name = "numbers"

    def __init__(self, language):
        self._language = language

    def __call__(self, text):
        return self._language.expand_numbers(text)


class StressStage:
    name = "stress"

    def __init__(self, language):
        self._language = language

    def __call__(self, text):
        return self._language.apply_stress(text)


class TokenizeStage:
    """Stressed text -> ids; optional sos/eos and unknown-char policy."""

    name = "tokenize"

    def __init__(self, language, sos_token="|", eos_token="~",
                 errors="filter"):
        self._language = language
        self._sos = sos_token
        self._eos = eos_token
        self._errors = errors

    def __call__(self, text, symbol_to_id):
        ids = self._language.tokenize(text, symbol_to_id,
                                      sos_token=self._sos,
                                      eos_token=self._eos,
                                      errors=self._errors)
        return add_sos_eos(ids, symbol_to_id, self._sos, self._eos)


__all__ = ["NormalizeStage", "AbbrevStage", "NumberStage", "StressStage",
           "TokenizeStage"]
