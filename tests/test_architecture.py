"""Architecture contract tests: languages, versions, factory, capabilities.

Torch-free and model-free: every test uses stubs or declarative layers.
"""
import ast
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

assert "torch" not in sys.modules, "test process imported torch before package!"

from silero_onnx.engine import EngineFactory  # noqa: E402
from silero_onnx.frontend import (AbstractTextFrontend, FrontendRegistry,  # noqa: E402
                                  TextFrontend, TextFrontend_Ru, add_sos_eos,
                                  load_symbol_table, tokenize)
from silero_onnx.languages import (AbstractLanguage, Language_Ru,  # noqa: E402
                                   LanguageRegistry)
from silero_onnx.models import (MODEL_SPECS, AbstractModelVersion,  # noqa: E402
                                ModelRegistry, Version_V5Ru, Version_V56)
from silero_onnx.models.spec import ModelSet, ModelSpec  # noqa: E402
from silero_onnx.strategies import (AbstractTTSStrategy, StrategyFactory,  # noqa: E402
                                    Strategy_V5, Strategy_V56, Strategy_V5_6)
from silero_onnx.strategies.common import (BreakProcessor, FeedBuilder,  # noqa: E402
                                           PauseService)
from silero_onnx.common.ssml_lowering import (  # noqa: E402
    SsmlLoweringService)
from silero_onnx.strategies.v5_6 import Feeds_V56, V56Feeds  # noqa: E402


class _TestFrontend:
    speakers = {"aidar": 0, "baya": 1, "kseniya": 2, "eugene": 3,
                "xenia": 4}

    @property
    def symbol_to_id(self):
        return load_symbol_table()[1]

    def text_to_ids(self, text):
        from silero_onnx.frontend import add_sos_eos, normalize, tokenize
        t = normalize(text)
        table = self.symbol_to_id
        return add_sos_eos(tokenize(t, table, errors="filter"), table), t

    def prepare(self, text, speaker="xenia"):
        ids, _stressed = self.text_to_ids(text)
        n = len(ids)
        return {
            "sequence": np.array([ids], dtype=np.int64),
            "speaker_ids": np.array([self.speakers[speaker]],
                                    dtype=np.int64),
            "durs_rate": np.ones((1, n), dtype=np.float32),
            "pitch_coefs": np.ones((1, n), dtype=np.float32),
        }


class Language_NoStress(AbstractLanguage):
    """Test double: a language without lexical stress."""

    language_id = "xx"
    requires_stress = False

    def __init__(self, **kwargs):
        pass

    def normalize(self, text):
        return " ".join(text.split())

    def get_symbol_table(self):
        return ["|", "~", " ", "а"], {"|": 0, "~": 1, " ": 2, "а": 3}

    def get_speakers(self):
        return {"voice": 0}


class Version_NoStress(AbstractModelVersion):
    """Test double: a version that declares no stress/SSML support."""

    version_id = "test_no_stress"
    _spec = ModelSpec(
        version_id="test_no_stress",
        display_name="test no-stress version",
        model_set=ModelSet(
            model_files={},
            session_topology=(),
            acoustic_inputs=("sequence", "speaker_ids", "durs_rate",
                             "focus_mask"),
        ),
        supports_stress=False,
        supports_ssml=False,
        pause_sil_value=-5.0,
    )

    @property
    def spec(self):
        return self._spec


def test_registries_and_naming_contract():
    assert LanguageRegistry.list() == ["ru"]
    assert isinstance(LanguageRegistry.get("ru"), Language_Ru)
    assert ModelRegistry.get("v5_ru").__class__ is Version_V5Ru
    assert ModelRegistry.get("v5_6_ru").__class__ is Version_V56
    assert StrategyFactory.get("v5_ru") is Strategy_V5
    assert StrategyFactory.get("v5_6_ru") is Strategy_V56
    assert Strategy_V5_6 is Strategy_V56  # historical spelling


def test_shared_services_and_feed_naming():
    from silero_onnx.strategies.common import ssml_lowering as legacy_ssml

    assert issubclass(PauseService, BreakProcessor)
    assert Feeds_V56 is V56Feeds
    assert Feeds_V56.__name__ == "Feeds_V56"
    assert (SsmlLoweringService.__module__
            == "silero_onnx.common.ssml_lowering")
    assert legacy_ssml.SsmlLoweringService is SsmlLoweringService


def test_language_abstraction_and_registry_extension():
    lang = Language_Ru()
    assert isinstance(lang, AbstractLanguage)
    assert lang.language_id == "ru"
    assert lang.requires_stress is True
    assert lang.normalize("ул. Ленина, 5") == "улица Ленина, пять"
    assert lang.get_symbol_table()[1]
    assert lang.get_speakers()

    # A language without stress gets the base no-op implementation.
    no_stress = Language_NoStress()
    assert no_stress.apply_stress("привет") == "привет"

    class Language_Test(Language_NoStress):
        language_id = "test_language_extension"

    LanguageRegistry.register(Language_Test)
    try:
        assert isinstance(LanguageRegistry.get("test_language_extension"),
                          Language_Test)
        with pytest.raises(ValueError, match="already registered"):
            LanguageRegistry.register(Language_Test)
    finally:
        LanguageRegistry.unregister("test_language_extension")

    with pytest.raises(ValueError, match="unknown language"):
        LanguageRegistry.get("test_language_extension")


def test_version_capabilities_are_declarative():
    v5 = ModelRegistry.get("v5_ru")
    v56 = ModelRegistry.get("v5_6_ru")
    assert v5.supports_stress and v5.supports_ssml and v5.supports_pitch
    assert v56.supports_stress and v56.supports_ssml and v56.supports_pitch
    assert v5.supports_questions is False
    assert v56.supports_questions is True
    assert v56.pause_sil_value == -5.0 and v5.pause_sil_value == -8.0


def test_model_version_interface_methods():
    v5 = ModelRegistry.get("v5_ru")
    v56 = ModelRegistry.get("v5_6_ru")

    assert v5.model_files()["acoustic"] == "acoustic_v5_ru.onnx"
    assert v5.session_topology() == [
        "stress", "ngram", "acoustic", "vocoder", "istft"]
    assert v5.has_pitch_coefs_input() is True
    assert v56.has_pitch_coefs_input() is True
    assert v56.requires_gt_durs() is True


def test_model_registry_extension_and_duplicate_guard():
    class Version_Test(Version_NoStress):
        version_id = "test_model_extension"
        _spec = ModelSpec(
            version_id=version_id,
            display_name="test extension version",
            model_set=ModelSet(model_files={}, session_topology=(),
                               acoustic_inputs=()),
            supports_stress=False,
            supports_ssml=False,
        )

    version = Version_Test()
    ModelRegistry.register(version)
    try:
        assert ModelRegistry.get("test_model_extension") is version
        with pytest.raises(ValueError, match="already registered"):
            ModelRegistry.register(Version_Test())
    finally:
        ModelRegistry.unregister("test_model_extension")

    with pytest.raises(ValueError, match="unknown model version"):
        ModelRegistry.get("test_model_extension")


def test_strategy_base_validates_requests_and_subclasses_contract():
    assert issubclass(Strategy_V5, AbstractTTSStrategy)
    assert issubclass(Strategy_V56, AbstractTTSStrategy)

    class Strategy_Test(AbstractTTSStrategy):
        version_id = "test_strategy_extension"

        def _synthesize(self, text, speaker, ssml=False):
            return "ok", None

    st = Strategy_Test("m", "d", {}, _TestFrontend(),
                       version=Version_NoStress())
    assert st.synthesize("раз", speaker="xenia") == ("ok", None)

    with pytest.raises(SystemExit, match="empty text"):
        st.synthesize("   ")
    with pytest.raises(SystemExit, match="sample_rate"):
        st.synthesize("раз", sample_rate=8000)
    with pytest.raises(ValueError, match="unknown speaker"):
        st.synthesize("раз", speaker="unknown")
    with pytest.raises(ValueError, match="SSML is not supported"):
        st.synthesize("<speak>раз</speak>", ssml=True)


def test_language_without_stress_is_noop_and_frontend_keeps_plain_text():
    lang = Language_NoStress()
    assert lang.apply_stress("привет") == "привет"
    fe = TextFrontend(lang)
    ids, stressed = fe.text_to_ids("  привет   мир ")
    assert stressed == "привет мир"  # no '+' inserted
    assert ids[0] == fe.symbol_to_id["|"] and ids[-1] == fe.symbol_to_id["~"]
    assert "*" not in stressed


def test_engine_factory_rejects_stress_incompatible_pair_before_io():
    ModelRegistry.register(Version_NoStress())
    try:
        with pytest.raises(ValueError, match="requires stress"):
            EngineFactory.create_engine("ru", Version_NoStress.version_id,
                                        model_dir="does-not-matter")
    finally:
        ModelRegistry.unregister(Version_NoStress.version_id)


def test_engine_factory_rejects_version_without_strategy_before_io():
    class Version_NoStrategy(AbstractModelVersion):
        version_id = "v6_ru"
        _spec = ModelSpec(
            version_id="v6_ru",
            display_name="registered without a strategy",
            model_set=ModelSet(model_files={}, session_topology=(),
                               acoustic_inputs=()),
            supports_stress=True,
            supports_ssml=True,
        )

        @property
        def spec(self):
            return self._spec

    ModelRegistry.register(Version_NoStrategy())
    try:
        with pytest.raises(
                ValueError, match=r"^no strategy for version v6_ru$"):
            EngineFactory.create_engine("ru", "v6_ru",
                                        model_dir="does-not-matter")
    finally:
        ModelRegistry.unregister("v6_ru")


def test_ssml_unsupported_version_raises_value_error():
    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=Version_NoStress())
    with pytest.raises(ValueError, match="SSML is not supported"):
        st.synthesize("<speak>раз два.</speak>", ssml=True)


def test_no_stress_version_gets_zero_focus_mask():
    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=Version_NoStress())
    feeds = st._prepare("раз два.", "xenia")
    assert feeds.focus_mask.shape == feeds.sequence.shape
    assert not feeds.focus_mask.any()
    assert set(np.unique(feeds.focus_mask)) == {0}


def test_strategy_v5_never_applies_question_classification(monkeypatch):
    from silero_onnx.strategies.v5_6 import emphasis as E

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Strategy_V5 must not classify questions")

    monkeypatch.setattr(E, "classify_sentence_type", unexpected)
    monkeypatch.setattr(E, "classify_sentence_types", unexpected)
    monkeypatch.setattr(E, "build_sentence_type_ids", unexpected)

    st = Strategy_V5("m", "d", {}, _TestFrontend(),
                     version=ModelRegistry.get("v5_ru"))
    feeds = st._prepare("Кто здесь?", "xenia")

    assert st.version.supports_questions is False
    assert feeds["sequence"].shape[1] > 2
    assert "type_ids" not in feeds


def test_strategy_v56_uses_question_classification(monkeypatch):
    from silero_onnx.strategies.v5_6 import emphasis as E

    calls = []
    original = E.classify_sentence_types

    def spy(text, **kwargs):
        calls.append((text, kwargs))
        return original(text, **kwargs)

    monkeypatch.setattr(E, "classify_sentence_types", spy)
    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=ModelRegistry.get("v5_6_ru"))
    feeds = st._prepare("Кто здесь?", "xenia")

    assert calls == [("Кто здесь?", {"classify_questions": True})]
    assert (int(feeds.type_ids[0, 1])
            == E.SENTENCE_TYPE_TO_ID[E.WH_QUESTION])


def test_v56_pitch_channel_is_declared_and_fed():
    v56 = ModelRegistry.get("v5_6_ru")
    assert v56.supports_pitch is True
    assert v56.has_pitch_coefs_input() is True
    assert "pitch_coefs" in v56.acoustic_inputs()

    st = Strategy_V56("m", "d", {}, _TestFrontend(), version=v56)
    feeds = st._prepare("Привет.", "xenia")

    assert feeds.pitch_coefs.shape == feeds.sequence.shape
    assert np.allclose(feeds.pitch_coefs, 1.0)


def test_strategy_v56_without_question_capability_uses_statement_type():
    from silero_onnx.strategies.v5_6 import emphasis as E

    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=Version_NoStress())
    feeds = st._prepare("Кто здесь?", "xenia")

    assert (int(feeds.type_ids[0, 1])
            == E.SENTENCE_TYPE_TO_ID[E.STATEMENT])


def test_no_stress_ssml_keeps_zero_focus_mask_and_breaks():
    class Version_NoStressSSML(Version_NoStress):
        version_id = "test_no_stress_ssml"
        _spec = ModelSpec(
            version_id=version_id,
            display_name="test no-stress SSML version",
            model_set=ModelSet(
                model_files={},
                session_topology=(),
                acoustic_inputs=("sequence", "speaker_ids", "durs_rate",
                                 "gt_durs", "gt_pitch", "type_ids",
                                 "focus_mask"),
            ),
            supports_stress=False,
            supports_ssml=True,
            requires_gt_durs=True,
            requires_gt_pitch=True,
        )

    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=Version_NoStressSSML())
    feeds, breaks = st._prepare_ssml(
        "<speak>раз <break time='1s'/> два</speak>", "xenia")

    assert not feeds.focus_mask.any()
    assert feeds.focus_mask.shape == feeds.sequence.shape
    assert feeds.durs_rate.shape == feeds.sequence.shape
    assert breaks


def test_emphasis_markers_rejected_without_stress_support():
    st = Strategy_V56("m", "d", {}, _TestFrontend(),
                      version=Version_NoStress())
    with pytest.raises(ValueError, match="emphasis markers"):
        st._prepare("*раз* два.", "xenia")


def test_feed_builder_validates_durs_rate_and_stress_inputs():
    fb = FeedBuilder(Version_NoStress())
    feeds = fb.acoustic_feeds(
        sequence=np.array([[1, 2, 1]], dtype=np.int64),
        speaker_ids=np.array([0], dtype=np.int64),
        durs_rate=np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    assert not feeds["focus_mask"].any()
    with pytest.raises(ValueError, match="durs_rate"):
        fb.acoustic_feeds(
            sequence=np.array([[1, 2, 1]], dtype=np.int64),
            speaker_ids=np.array([0], dtype=np.int64),
            durs_rate=np.array([[1.0, 0.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_feed_builder_rejects_invalid_durs_rate(bad):
    fb = FeedBuilder(Version_NoStress())
    with pytest.raises(ValueError, match="durs_rate"):
        fb.acoustic_feeds(
            sequence=np.array([[1, 2, 1]], dtype=np.int64),
            speaker_ids=np.array([0], dtype=np.int64),
            durs_rate=np.array([[1.0, bad, 1.0]], dtype=np.float32))


def test_ssml_lowering_rejects_nonpositive_rate():
    service = SsmlLoweringService(normalize=lambda t: t)
    with pytest.raises(ValueError, match="durs_rate"):
        service.plan("<speak><prosody rate='0%'>раз</prosody></speak>")


def test_ssml_lowering_service_attributes_rates_and_warns_without_pitch():
    service = SsmlLoweringService(normalize=lambda text: text)
    plan = service.plan(
        "<speak><prosody rate='slow'>раз</prosody> два</speak>")
    assert [s.text for s in plan.segments] == ["раз", "два"]
    assert plan.segments[0].rate == pytest.approx(0.8)

    table = load_symbol_table()[1]
    stressed = " ".join(s.text for s in plan.segments)
    ids = add_sos_eos(tokenize(stressed, table, errors="filter"), table)
    rates, breaks = service.attribute(
        stressed, ids, table, plan.segments, plan.leading_break_ms)

    assert len(rates) == len(ids)
    assert rates[0] == pytest.approx(0.8)
    assert rates[-1] == pytest.approx(1.0)
    assert breaks == []

    no_pitch = SsmlLoweringService(
        normalize=lambda text: text,
        supports_pitch=False,
        version_id="test_no_pitch",
    )
    with pytest.warns(UserWarning, match="ignores <prosody pitch"):
        plan_no_pitch = no_pitch.plan(
            "<speak><prosody pitch='high'>раз</prosody></speak>")
    assert plan_no_pitch.requested_pitches == (1.15,)


def test_text_frontend_ru_uses_injected_language():
    lang = Language_Ru()
    fe = TextFrontend_Ru(language=lang)
    assert fe.language is lang
    assert isinstance(fe, AbstractTextFrontend)


def test_frontend_registry_routes_by_language_and_accepts_extensions():
    assert FrontendRegistry.list() == ["ru"]
    fe = FrontendRegistry.get("ru", language=Language_NoStress())
    assert isinstance(fe, TextFrontend_Ru)
    assert isinstance(FrontendRegistry.get(
        "xx", default=TextFrontend, language=Language_NoStress()),
        TextFrontend)

    class Frontend_Test(TextFrontend):
        frontend_id = "test_frontend_extension"

    FrontendRegistry.register(Frontend_Test)
    try:
        got = FrontendRegistry.get("test_frontend_extension",
                                   language=Language_NoStress())
        assert type(got) is Frontend_Test
        with pytest.raises(ValueError, match="already registered"):
            FrontendRegistry.register(Frontend_Test)
    finally:
        FrontendRegistry.unregister("test_frontend_extension")

    with pytest.raises(ValueError, match="unknown frontend"):
        FrontendRegistry.get("test_frontend_extension")


def test_frontend_emits_pitch_only_when_capability_is_enabled():
    lang = Language_NoStress()
    no_pitch = TextFrontend(lang, supports_pitch=False)
    assert "pitch_coefs" not in no_pitch.prepare("а", "voice")

    with pytest.warns(UserWarning, match="ignores <prosody pitch"):
        feeds, breaks = no_pitch.prepare_ssml(
            "<speak><prosody pitch='high'>а</prosody></speak>", "voice")
    assert "pitch_coefs" not in feeds
    assert breaks == []

    with_pitch = TextFrontend(lang, supports_pitch=True)
    plain = with_pitch.prepare("а", "voice")
    assert np.allclose(plain["pitch_coefs"], 1.0)
    pitched, breaks = with_pitch.prepare_ssml(
        "<speak><prosody pitch='high'>а</prosody></speak>", "voice")
    assert np.allclose(pitched["pitch_coefs"], 1.15)
    assert breaks == []


def test_legacy_frontend_surface_remains_available():
    from silero_onnx import frontend as F
    from silero_onnx.versions import (DEFAULT_VERSION, MODEL_SETS,
                                      VERSIONS)

    table = load_symbol_table()[1]
    assert F.prepare("Привет, 5", table, stress=False)
    legacy = F.TorchFreeFrontend()
    assert legacy.speakers
    assert legacy.symbol_to_id == table
    for name in ("normalize", "prepare", "tokenize", "add_sos_eos",
                 "apply_stress", "get_accentor", "load_symbol_table",
                 "load_speaker_map", "DATA_DIR"):
        assert hasattr(F, name), name

    assert DEFAULT_VERSION == "v5_6_ru"
    assert set(MODEL_SETS) == set(VERSIONS) == {"v5_ru", "v5_6_ru"}


def test_runtime_import_graph_is_torch_free():
    import importlib

    for name in ("silero_onnx", "silero_onnx.engine",
                 "silero_onnx.frontend", "silero_onnx.languages",
                 "silero_onnx.models", "silero_onnx.strategies"):
        importlib.import_module(name)
    assert "torch" not in sys.modules

    # The frontend layer must not depend on strategies (even lazily):
    # version-neutral SSML lowering lives in silero_onnx.common.
    frontend_dir = os.path.join(ROOT, "silero_onnx", "frontend")
    for name in sorted(os.listdir(frontend_dir)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(frontend_dir, name), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "strategies" not in (node.module or ""), (
                    name, node.module)


def test_check_models_acoustic_io_is_derived_from_model_specs():
    import importlib.util

    path = os.path.join(ROOT, "scripts", "check_models.py")
    spec = importlib.util.spec_from_file_location("check_models_contract",
                                                   path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from silero_onnx.versions import MODEL_SETS

    for version in MODEL_SPECS.values():
        acoustic = version.model_files["acoustic"]
        assert module.EXPECTED[acoustic][0] == list(version.acoustic_inputs)
    artifacts = {f for files in MODEL_SETS.values() for f in files.values()}
    assert artifacts <= set(module.EXPECTED)


def test_ort_session_policy_is_reproducible():
    import onnxruntime as ort
    from silero_onnx.ort import session_options
    so = session_options()
    assert so.intra_op_num_threads == 2
    assert so.inter_op_num_threads == 1
    assert so.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


def _classes(relpath):
    path = os.path.join(ROOT, relpath)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {node.name: [b.id if isinstance(b, ast.Name) else
                        getattr(b, "attr", "") for b in node.bases]
            for node in tree.body if isinstance(node, ast.ClassDef)}


def test_export_tracks_use_the_shared_template_method():
    base = _classes("scripts/strategies/base.py")
    assert "AbstractExportPipeline" in base
    v5 = _classes("scripts/strategies/v5/export.py")
    assert v5["ExportPipeline_V5"] == ["AbstractExportPipeline"]
    v56 = _classes("scripts/strategies/v5_6/export.py")
    assert v56["ExportPipeline_V56"] == ["AbstractExportPipeline"]
    acoustic = _classes("scripts/strategies/v5_6/acoustic.py")
    assert {"Acoustic_V56", "Tacotron_V56"} <= set(acoustic)
    assert all(name.endswith("_V56") for name in acoustic)
    for relpath in (
            "scripts/strategies/v5_6/forward.py",
            "scripts/strategies/v5_6/series.py",
            "scripts/strategies/v5_6/vocoder.py"):
        classes = set(_classes(relpath))
        assert classes and all(name.endswith("_V56") for name in classes), (
            relpath, sorted(classes))
