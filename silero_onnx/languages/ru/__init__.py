"""Russian language implementation (``Language_Ru``)."""
from ..base import AbstractLanguage, collapse_whitespace
from .abbreviations import expand_abbreviations
from .numbers import expand_numbers
from .stress import OnnxStressProvider
from .symbols import DATA_DIR, load_speaker_map, load_symbol_table


class Language_Ru(AbstractLanguage):
    """Russian: Cyrillic symbols, abbreviations, numbers, ONNX stress."""

    language_id = "ru"
    requires_stress = True

    def __init__(self, models_dir=None, data_dir=None):
        self._models_dir = models_dir
        self._data_dir = data_dir or DATA_DIR
        self._symbols, self._symbol_to_id = load_symbol_table(self._data_dir)
        self._speakers = load_speaker_map(self._data_dir)
        self._stress = OnnxStressProvider(models_dir=models_dir,
                                          data_dir=self._data_dir)

    def normalize(self, text: str) -> str:
        """Whitespace, abbreviations, numbers (no ML, deterministic)."""
        text = collapse_whitespace(text)
        return expand_numbers(expand_abbreviations(text))

    def expand_abbreviations(self, text: str) -> str:
        return expand_abbreviations(text)

    def expand_numbers(self, text: str) -> str:
        return expand_numbers(text)

    def apply_stress(self, text: str) -> str:
        if not self.requires_stress:
            return text
        return self._stress(text)

    def get_symbol_table(self):
        return list(self._symbols), dict(self._symbol_to_id)

    def get_speakers(self):
        return dict(self._speakers)

__all__ = ["Language_Ru", "DATA_DIR"]
