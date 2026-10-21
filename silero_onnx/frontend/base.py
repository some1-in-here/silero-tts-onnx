"""Abstract text-frontend contract (version-independent)."""
from abc import ABC, abstractmethod


class AbstractTextFrontend(ABC):
    """Turns language-specific text into model inputs.

    The frontend receives an ``AbstractLanguage`` through its
    constructor (dependency injection) and knows nothing about model
    versions or ONNX sessions.
    """

    frontend_id: str = ""

    def __init__(self, language):
        self._language = language

    @property
    def language(self):
        return self._language

    @property
    @abstractmethod
    def speakers(self):
        """Voice name -> embedding id (or an ordered name list)."""

    @property
    @abstractmethod
    def symbol_to_id(self):
        """Symbol -> id table."""

    @abstractmethod
    def text_to_ids(self, text):
        """Text -> (ids with sos/eos, stressed text)."""

    @abstractmethod
    def prepare(self, text, speaker="xenia"):
        """Plain text -> acoustic feed dict."""

    @abstractmethod
    def prepare_ssml(self, ssml_text, speaker="xenia"):
        """SSML -> (acoustic feed dict, break plans)."""


__all__ = ["AbstractTextFrontend"]
