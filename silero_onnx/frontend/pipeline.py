"""TextPipeline: ordered composition of normalization/stress/tokenize stages."""
from .stages import (AbbrevStage, NormalizeStage, NumberStage, StressStage,
                     TokenizeStage)


class TextPipeline:
    """normalize -> abbreviations -> numbers -> stress -> tokenize.

    The first three stages can be replaced/augmented for a new language;
    stress and tokenize always come from the injected language object.
    """

    def __init__(self, language, stages=None):
        self._language = language
        self.stages = tuple(stages) if stages is not None else (
            NormalizeStage(),
            AbbrevStage(language),
            NumberStage(language),
        )
        self._stress_stage = StressStage(language)
        self._tokenize_stage = TokenizeStage(language)

    def normalize(self, text):
        for stage in self.stages:
            text = stage(text)
        return text

    def stress(self, text):
        return self._stress_stage(text)

    def tokenize(self, text, symbol_to_id):
        return self._tokenize_stage(text, symbol_to_id)

    def text_to_ids(self, text, symbol_to_id):
        normalized = self.normalize(text)
        stressed = self.stress(normalized)
        return self.tokenize(stressed, symbol_to_id), stressed


__all__ = ["TextPipeline"]
