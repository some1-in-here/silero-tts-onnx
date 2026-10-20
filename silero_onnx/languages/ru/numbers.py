"""Russian number expansion (moved from frontend.py unchanged)."""
import re

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Minus as numeric sign only (not hyphenation): not preceded by a
# digit/dot/comma/word char, followed by a digit. "5-10" keeps its hyphen.
_MINUS_RE = re.compile(r"(?<![\d.,\w])-(?=\d)")


def _num_to_words(token: str) -> str:
    # NOTE: masculine nominative cardinal only ("одна книга", prepositional
    # "в ... году" need gender/case agreement — a known hard problem, out of
    # prototype scope; output stays grammatical for the common cases).
    # Digits have NO symbols in the v5 table, so unexpanded numbers would
    # fail tokenize() — this stage is required, not optional.
    from num2words import num2words as n2w
    t = token.replace(",", ".")
    try:
        if "." in t:
            whole, frac = t.split(".")
            w = n2w(int(whole), lang="ru")
            digits = {"0": "ноль", "1": "один", "2": "два", "3": "три",
                      "4": "четыре", "5": "пять", "6": "шесть", "7": "семь",
                      "8": "восемь", "9": "девять"}
            return w + " целых " + " ".join(digits[d] for d in frac
                                            if d.isdigit())
        return n2w(int(t), lang="ru")
    except Exception:
        return token  # leave untouched on any failure (loud fallback below)


def expand_numbers(text: str) -> str:
    text = _MINUS_RE.sub("минус ", text)
    return _NUM_RE.sub(lambda m: _num_to_words(m.group(0)), text)
