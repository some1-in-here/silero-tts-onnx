"""Factory routing + v5_6 feed contract (torch-free, no models)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

assert "torch" not in sys.modules, "test process imported torch before package!"

from silero_onnx import env  # noqa: E402

from silero_onnx.strategies import _STRATEGIES, create_strategy  # noqa: E402
from silero_onnx.strategies.v5 import Strategy_V5  # noqa: E402
from silero_onnx.strategies.v5_6 import Strategy_V5_6  # noqa: E402
from silero_onnx.versions import DEFAULT_VERSION, VERSIONS  # noqa: E402


def test_registry_covers_versions():
    assert set(_STRATEGIES) == set(VERSIONS) == {"v5_ru", "v5_6_ru"}
    assert DEFAULT_VERSION == "v5_6_ru"


def test_routing_classes():
    assert _STRATEGIES["v5_ru"] is Strategy_V5
    assert _STRATEGIES["v5_6_ru"] is Strategy_V5_6


def test_unknown_version_value_error():
    with pytest.raises(ValueError):
        create_strategy(version="v9")


class _Frontend:
    speakers = {"aidar": 0, "baya": 1, "kseniya": 2, "eugene": 3,
                "xenia": 4}

    @property
    def symbol_to_id(self):
        from silero_onnx.frontend import load_symbol_table
        return load_symbol_table()[1]

    def text_to_ids(self, text):
        from silero_onnx.frontend import (add_sos_eos, normalize, tokenize)
        t = normalize(text)
        table = self.symbol_to_id
        return add_sos_eos(tokenize(t, table, errors="filter"), table), t


def test_prepare_field_contract():
    """_prepare field contract: base == ti % 6, fm in {0,3}, rate ones.

    Named fields (V56Feeds) make an fm/base swap unconstructable by
    callers (an ORT Gather OOB is what a swap costs)."""
    import numpy as np
    st = Strategy_V5_6("m", "d", {}, _Frontend())
    f = st._prepare("Меня зовут *Ксения*.", "xenia")
    assert f.sequence.dtype == np.int64 and f.speaker_ids.tolist() == [4]
    assert (f.base_ids == f.type_ids % 6).all()
    assert set(np.unique(f.focus_mask)) <= {0, 3}
    assert f.durs_rate.shape == f.sequence.shape
    assert (f.durs_rate == 1.0).all()


def test_new_upstream_voice_needs_no_code_changes():
    """An extra voice in the frontend table must not break init.

    New voice = new embedding row = new model files + updated
    speakers.json (data, not code). Nothing else changes: ids stay
    dense, listing reads the same table.
    """
    class Extra(_Frontend):
        speakers = dict(_Frontend.speakers, zahar=5)

    Strategy_V5_6("m", "d", {}, Extra())  # must not raise


def test_non_dense_speaker_ids_fail_loud():
    """Speaker ids must be dense 0..N-1 (gap = corrupt table)."""

    class Gapped(_Frontend):
        speakers = dict(_Frontend.speakers, zahar=9)

    with pytest.raises(SystemExit):
        Strategy_V5_6("m", "d", {}, Gapped())


def test_empty_speaker_name_fails_loud():
    """Empty voice names mean a corrupt table."""

    class Nameless(_Frontend):
        speakers = {**_Frontend.speakers, "": 5}

    with pytest.raises(SystemExit):
        Strategy_V5_6("m", "d", {}, Nameless())


def test_library_requires_explicit_model_dir(monkeypatch):
    """No <repo>/models/onnx autodiscovery in library code (CLI-only privilege).

    With model_dir=None and no env var, the factory must fail loudly
    demanding an explicit path instead of silently using the build output.
    """
    from silero_onnx.strategies import SileroTTS

    monkeypatch.delenv(env.MODELS_DIR_VAR, raising=False)
    try:
        SileroTTS(model_dir=None)
    except SystemExit as e:
        assert "explicitly" in str(e.code), str(e.code)
    else:
        raise AssertionError("bare SileroTTS() must fail without model_dir")
