"""Language abstraction (version-independent).

A language owns text normalization, abbreviation/number expansion,
stress marking, the symbol table and the speaker/voice table. It never
imports a model version or an ONNX session; the frontend injects it.

Naming contract: language classes are ``Language_<Id>`` (e.g.
``Language_Ru``) — entity first, version/locale suffix after an
underscore.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import re


class AbstractLanguage(ABC):
    """Interface every language implementation must satisfy."""

    #: Stable identifier used by LanguageRegistry and EngineFactory.
    language_id: str = ""
    #: True when the language needs lexical stress (Russian).
    #: False means apply_stress() is a no-op and focus masks stay zero.
    requires_stress: bool = False

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Whitespace cleanup + abbreviations + number expansion."""

    def expand_abbreviations(self, text: str) -> str:
        """Optional hook used by TextPipeline's AbbrevStage."""
        return text

    def expand_numbers(self, text: str) -> str:
        """Optional hook used by TextPipeline's NumberStage."""
        return text

    def apply_stress(self, text: str) -> str:
        """Add stress marks / resolve homographs.

        Languages with ``requires_stress = False`` inherit this no-op:
        no ``+`` is inserted and no exception is raised.
        """
        return text

    @abstractmethod
    def get_symbol_table(self) -> Tuple[List[str], Dict[str, int]]:
        """Return (ordered symbols, symbol -> id)."""

    @abstractmethod
    def get_speakers(self) -> Dict[str, int]:
        """Return voice name -> embedding id."""

    def tokenize(self, text: str, symbol_to_id: Dict[str, int],
                 sos_token: str = "|", eos_token: str = "~",
                 add_sos_eos: bool = False, errors: str = "raise") -> List[int]:
        """Map text to symbol ids (language-independent tokenization).

        Kept on the language so a future language with a different
        alphabet can override it without touching the frontend.
        """
        return tokenize(text, symbol_to_id, sos_token=sos_token,
                        eos_token=eos_token, add_sos_eos=add_sos_eos,
                        errors=errors)


def tokenize(text: str, symbol_to_id: Dict[str, int],
             sos_token: str = "|", eos_token: str = "~",
             add_sos_eos: bool = False, errors: str = "raise") -> List[int]:
    """Map cleaned+stressed text to symbol ids.

    Space is a regular symbol and is mapped (not dropped). Unknown
    characters: ``errors="raise"`` (default, loud) or ``"filter"``
    (drop silently, like the original pipeline's regex).
    """
    seq = text.lower().replace("—", "–")
    ids, unknown = [], []
    for i, ch in enumerate(seq):
        if ch in symbol_to_id:
            ids.append(symbol_to_id[ch])
        elif errors == "filter":
            continue
        else:
            unknown.append((i, ch))
    if unknown:
        raise ValueError(f"unknown symbols at {unknown[:8]} (use "
                         f"errors='filter' for original-like silent dropping)")
    if add_sos_eos:
        ids = [symbol_to_id[sos_token]] + ids + [symbol_to_id[eos_token]]
    return ids


def add_sos_eos(ids: List[int], symbol_to_id: Dict[str, int],
                sos_token: str = "|", eos_token: str = "~") -> List[int]:
    return [symbol_to_id[sos_token]] + list(ids) + [symbol_to_id[eos_token]]


def collapse_whitespace(text: str) -> str:
    """Deterministic whitespace cleanup (the NormalizeStage core)."""
    return re.sub(r"\s+", " ", text).strip()
