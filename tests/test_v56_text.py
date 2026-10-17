"""v5_6 text algorithms: exact vectors (torch-free, no models).

Every vector below was verified against the torch v5_6_ru package
functions (parity probe: classify/spans/ti/fm exact on 12 texts incl.
**/* combos). These tests pin the port, not the probe.
"""
import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

assert "torch" not in sys.modules, "test process imported torch before package!"

from silero_onnx.strategies.v5_6 import emphasis as E  # noqa: E402
from silero_onnx.frontend import load_symbol_table, tokenize  # noqa: E402


def test_classify_taxonomy():
    assert E.classify_text("Кто здесь?") == "wh_q"
    assert E.classify_text("Меня зовут Ксения!") == "exclam"
    assert E.classify_text("Правда, да?") == "tag_q"
    assert E.classify_text("Ты будешь чай или кофе?") == "alternative_q"
    assert E.classify_text("Привет.") == "st"
    assert E.classify_text("Меня зовут Ксения?") == "general_q"
    assert E.classify_text("") == "st"
    assert E.classify_text("Привет. Пока!") == "st|exclam"


def test_split_emphasis():
    text, idx, has = E.split_emphasis("Меня зовут **Ксения**.")
    assert (text, idx, has) == ("Меня зовут *Ксения*.", {0}, True)
    text, idx, has = E.split_emphasis("Меня зовут Ксения.")
    assert (text, idx, has) == ("Меня зовут Ксения.", set(), False)
    with pytest.raises(ValueError):
        E.split_emphasis("Некорректная **разметка.")
    # ** wraps a SINGLE word (upstream regex [^*\s]+): spaces inside
    # are invalid markup and raise upstream's own message...
    with pytest.raises(ValueError, match="эмфазы"):
        E.split_emphasis("**Тихо говорю.**")
    # ...while per-word ** and multi-word * spans are fine
    text, idx, has = E.split_emphasis("**Тихо** **говорю.**")
    assert (text, idx, has) == ("*Тихо* *говорю.*", {0, 1}, True)


def test_extract_intensity():
    assert E.extract_intensity("просто текст") == ("просто текст", 3)
    assert E.extract_intensity("<intensity=1>тихо</intensity>") == ("тихо", 1)
    with pytest.raises(ValueError):
        E.extract_intensity("<intensity=9>громко</intensity>")
    # several tags: the first wins for the whole utterance (upstream
    # keeps intensities[0]); differing values warn instead of a silent
    # drop, equal ones warn not at all
    with pytest.warns(UserWarning, match="first one"):
        assert E.extract_intensity(
            "<intensity=1>а</intensity> <intensity=3>б</intensity>"
        ) == ("а б", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert E.extract_intensity(
            "<intensity=2>а</intensity> <intensity=2>б</intensity>"
        ) == ("а б", 2)


def test_place_tokens():
    assert E.place_tokens([0, 1, 2], 6).tolist() == [[0, 0, 1, 2, 0, 0]]
    assert E.place_tokens([], 4).tolist() == [[0, 0, 0, 0]]


def test_char_sent_ids():
    assert E.char_sent_ids("Привет! Как дела?") == [0] * 8 + [1] * 9


def test_locate_emph_sents():
    assert E.locate_emph_sents("Привет! Меня *зовут* *Ксения*.", {1}) == {1}
    assert E.locate_emph_sents("Привет! Меня зовут Ксения.", set()) == set()


def test_token_spans_and_type_ids():
    syms, table = load_symbol_table()
    ids = [table["|"]] + tokenize("приве+т! пока", table) + [table["~"]]
    spans, sents = E.token_sent_spans(ids, syms)
    assert spans == [(1, 9), (10, 14)]
    assert sents == ["приве+т!", "пока"]
    ti = E.build_type_ids_tokens(ids, ["exclam", "st"], {1}, syms)
    assert ti.tolist() == [[5] * 10 + [6] * 5]
    assert (ti % 6).tolist() == [[5] * 10 + [0] * 5]


def test_parse_focus():
    _syms, table = load_symbol_table()
    _clean, fm = E.parse_focus_from_text("*приве+т*! пока", 15, table,
                                         intensity=3)
    assert fm.tolist() == [[3] * 8 + [0] * 7]
    _clean, fm = E.parse_focus_from_text("привет", 8, table, intensity=3)
    assert fm.tolist() == [[0] * 8]
