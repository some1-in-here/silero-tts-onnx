"""Engine factory/orchestrator + CLI for torch-free Silero TTS.

Layer contract:
  EngineFactory.create_engine(language_id, version_id, model_dir)
    -> LanguageRegistry / ModelRegistry
    -> capability checks (stress, SSML)
    -> TextFrontend (language injected via DI)
    -> ONNX sessions for the version
    -> Strategy_* (sessions + frontend injected via DI)

The CLI lives here for backward compatibility; all inference is in
``strategies/`` and all text rules in ``languages/`` + ``frontend/``.
"""
import argparse
import os
import sys

import numpy as np  # noqa: F401 (console-side array handling)

from . import env
from .frontend import FrontendRegistry, TextFrontend, load_speaker_map
from .languages import LanguageRegistry
from .models import ModelRegistry
from .ort import create_session
from .strategies import StrategyFactory
from .versions import DEFAULT_VERSION, MODEL_SETS, VERSIONS

assert "torch" not in sys.modules, "torch must never be imported here"


# ---------------------------------------------------------------------
# Orchestrator / factory
# ---------------------------------------------------------------------

class TTSEngine:
    """A fully assembled engine: language + frontend + sessions + strategy."""

    def __init__(self, language, version, frontend, sessions, strategy):
        self.language = language
        self.version = version
        self.frontend = frontend
        self.sessions = sessions
        self.strategy = strategy

    @property
    def speakers(self):
        return self.strategy.speakers

    def synthesize(self, *args, **kwargs):
        return self.strategy.synthesize(*args, **kwargs)

    def synthesize_to_file(self, *args, **kwargs):
        return self.strategy.synthesize_to_file(*args, **kwargs)


class EngineFactory:
    """Abstract Factory: builds an engine from (language, model version)."""

    @classmethod
    def create_engine(cls, language_id="ru", version_id=DEFAULT_VERSION,
                      model_dir=None, data_dir=None):
        # 1. Resolve declarative layers first: no sessions are touched
        #    until the pair is proven compatible.
        try:
            language = LanguageRegistry.get(language_id,
                                            models_dir=model_dir,
                                            data_dir=data_dir)
        except (OSError, KeyError) as e:
            raise SystemExit(
                f"silero-onnx: error: cannot load frontend data for "
                f"language '{language_id}': {e}")
        version = ModelRegistry.get(version_id)
        if language.requires_stress and not version.supports_stress:
            raise ValueError(
                f"Language '{language_id}' requires stress, but version "
                f"'{version_id}' does not support it")
        if version_id not in StrategyFactory.list():
            raise ValueError(f"no strategy for version {version_id}")

        # 2. Resolve the model directory (library never auto-discovers
        #    <repo>/models/onnx; the CLI does that separately).
        if model_dir is None:
            model_dir = _env_models_dir(version_id)
        if model_dir is None:
            raise SystemExit(
                "silero-onnx: error: no model_dir given: pass "
                "model_dir='path/to/models' explicitly (or set "
                f"{env.MODELS_DIR_VAR}). The library never auto-discovers "
                f"<repo>/models/onnx/ - only the silero-onnx CLI does that "
                f"(version '{version_id}' needs: "
                f"{', '.join(version.model_files().values())})")
        if not os.path.isdir(model_dir):
            raise SystemExit(
                "silero-onnx: error: no ONNX models found for version "
                f"'{version_id}' (expected "
                f"{', '.join(version.model_files().values())} in model_dir). "
                "Build them with python scripts/export_all.py (v5_ru) or "
                "python scripts/strategies/v5_6/export.py (v5_6_ru) "
                "(see scripts/README.md), pass model_dir=... explicitly, or "
                f"set {env.MODELS_DIR_VAR}. ({env.MODELS_DIR_VAR}=%r, "
                "model_dir=%r)" % (env.models_dir(), model_dir))
        data_dir = data_dir or _dev_data_dir()

        # 3. Inject the language and the version's pitch capability into
        #    the frontend. The registry makes language-specific frontends
        #    additive; the generic frontend is the fallback.
        frontend = FrontendRegistry.get(
            language_id, default=TextFrontend, language=language,
            data_dir=data_dir, supports_pitch=version.supports_pitch)

        # 4. Open the version's sessions with the shared ORT policy.
        sessions = {}
        for key, fname in version.model_files().items():
            path = os.path.join(model_dir, fname)
            if not os.path.isfile(path):
                raise SystemExit(
                    f"silero-onnx: error: missing model file {path} "
                    f"(version '{version_id}' needs: "
                    f"{', '.join(version.model_files().values())})")
            try:
                sessions[key] = create_session(path)
            except SystemExit:
                raise
            except Exception as e:
                raise SystemExit(
                    f"silero-onnx: error: cannot load {path}: {e}")

        # 5. Inject frontend/sessions/version into the strategy.
        strategy = StrategyFactory.create(
            version_id, model_dir, data_dir, sessions, frontend,
            version=version, language=language)
        return TTSEngine(language, version, frontend, sessions, strategy)


class SileroTTS:
    """Version- and language-selected facade (defaults: ru, v5_6_ru)."""

    def __init__(self, model_dir=None, data_dir=None,
                 version=DEFAULT_VERSION, language="ru"):
        self._engine = EngineFactory.create_engine(
            language_id=language, version_id=version,
            model_dir=model_dir, data_dir=data_dir)

    @property
    def version(self):
        return self._engine.version.version_id

    @property
    def language(self):
        return self._engine.language.language_id

    @property
    def speakers(self):
        return self._engine.speakers

    def synthesize(self, *args, **kwargs):
        return self._engine.synthesize(*args, **kwargs)

    def synthesize_to_file(self, *args, **kwargs):
        return self._engine.synthesize_to_file(*args, **kwargs)


# ---------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------

def _env_models_dir(version=DEFAULT_VERSION):
    """SILERO_ONNX_MODELS env var, validated by the version's acoustic file."""
    acoustic = MODEL_SETS.get(version, {}).get("acoustic", "")
    envdir = env.models_dir()
    if envdir and acoustic and os.path.isfile(os.path.join(envdir, acoustic)):
        return envdir
    return None


def _cli_models_dir(version=DEFAULT_VERSION):
    """Model directory autoresolve — CLI ONLY (env var, then <repo>/models/onnx)."""
    found = _env_models_dir(version)
    if found is not None:
        return found
    acoustic = MODEL_SETS.get(version, {}).get("acoustic", "")
    here = os.path.dirname(os.path.abspath(__file__))
    repo_onnx = os.path.normpath(os.path.join(here, "..", "models", "onnx"))
    if acoustic and os.path.isfile(os.path.join(repo_onnx, acoustic)):
        return repo_onnx
    return None


def _dev_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------
# Console / CLI
# ---------------------------------------------------------------------

def _use_color():
    """ANSI colors only for live terminals that allow them."""
    return env.colors_enabled()


def _paint(code, s):
    return f"\033[{code}m{s}\033[0m" if _use_color() else s


class ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """argparse help with section/flag coloring (tty only, via _paint)."""

    def format_help(self):
        import re
        text = super().format_help()
        if not _use_color():
            return text
        text = re.sub(r"(?m)^usage:", "\033[33musage:\033[0m", text)
        text = re.sub(r"(?m)^(positional arguments|options|examples):",
                      "\033[33m\\1:\033[0m", text)
        text = re.sub(r"(--[a-z-]+)", "\033[32m\\1\033[0m", text)
        text = re.sub(r"\"[^\"]*\"", lambda m: "\033[36m" + m.group(0)
                      + "\033[0m", text)
        return text


# Quoted payloads for examples_text(): plain strings (no f-string
# backslash escapes) so the module parses on Python 3.10+. Every
# payload is self-describing: the spoken text itself demonstrates or
# narrates the feature of its example (no legend comments needed).
_EX_00 = '"Привет! Меня зовут Ксения."'
_EX_01 = '"Это версия по умолчанию. Вот слово с нажимом: **важно**."'
_EX_02 = '"Классическая версия: **звёздочки** здесь не работают."'
_EX_03 = '"Сначала с нажимом: **важно**. Потом без нажима: важно."'
_EX_04 = '"Сначала в фокусе: *важно*. Потом без фокуса: важно."'
_EX_05 = '"<intensity=1>Фокус слабее: *важно*.</intensity>"'
_EX_06 = '"<intensity=3>Фокус сильнее: *важно*.</intensity>"'
_EX_07 = '"Так звучит этот голос."'
_EX_08 = '"З+амок был открыт."'
_EX_09 = '"Зам+ок был открыт."'
_EX_10 = '"Подожди... Продолжаю."'
_EX_11 = '"Он — молодец."'
_EX_12 = '"<speak>Сейчас будет пауза. <break time=\'500ms\'/> Вот она была.</speak>"'
_EX_13 = '"<speak>Сейчас будет пауза. <break time=\'2s\'/> Вот она была.</speak>"'
_EX_14 = '"<speak>Сейчас будет пауза. <break strength=\'weak\'/> Вот она была.</speak>"'
_EX_15 = '"<speak>Первая часть. <break strength=\'x-strong\'/> Вторая часть после очень длинной паузы.</speak>"'
_EX_16 = '"<speak>Раз. <break time=\'250ms\'/> Два. <break strength=\'strong\'/> Три.</speak>"'
_EX_17 = '"<speak><prosody rate=\'fast\'>Я говорю быстро-быстро!</prosody> <prosody rate=\'slow\'>А теперь медленно и размеренно.</prosody>А теперь как обычно.</speak>"'
_EX_18 = '"<speak><prosody pitch=\'high\'>Здесь я говорю высоко.</prosody> <prosody pitch=\'low\'>А здесь низко.</prosody></speak>"'
_EX_19 = '"<speak><prosody rate=\'slow\' pitch=\'low\'>Говорю медленно и низко.</prosody></speak>"'
_EX_20 = '"<speak><prosody rate=\'slow\'>Говорю медленно, <prosody pitch=\'high\'>а это слово высоко</prosody> и снова ровно.</prosody></speak>"'
_EX_21 = '"<speak><s>Это конец первого предложения.</s><s>А это начало второго.</s></speak>"'
_EX_22 = '"Привет из указанной папки моделей!"'


def examples_text():
    """Structured, color-aware usage examples (see --examples)."""
    c = _use_color()

    def H(s):  # yellow: section headers
        return f"\033[33m{s}\033[0m" if c else s

    def F(s):  # green: flags
        return f"\033[32m{s}\033[0m" if c else s

    def Q(s):  # cyan: quoted text
        return f"\033[36m{s}\033[0m" if c else s

    def G(s):  # gray: group labels
        return f"\033[90m{s}\033[0m" if c else s

    def W(s):  # red: notes
        return f"\033[31m{s}\033[0m" if c else s

    return "\n".join([
        H("📌 Basic"),
        f"  silero-onnx {Q(_EX_00)} "
        f"{F('-o')} out.wav",
        "",
        H("🎯 Versions (default is v5_6_ru)"),
        f"  silero-onnx {Q(_EX_01)} "
        f"{F('-o')} v56.wav",
        f"  silero-onnx {F('--model-version')} v5_ru {Q(_EX_02)} "
        f"{F('-o')} v5.wav",
        f"  silero-onnx {F('--model-list')}",
        "",
        H("✨ Emphasis (**word** emphasized, *word* focused, "
          "<intensity=0..3> strength)"),
        f"  silero-onnx {Q(_EX_03)} "
        f"{F('-o')} emph.wav",
        f"  silero-onnx {Q(_EX_04)} "
        f"{F('-o')} focus.wav",
        f"  silero-onnx {Q(_EX_05)} "
        f"{F('-o')} focus1.wav",
        f"  silero-onnx {Q(_EX_06)} "
        f"{F('-o')} focus3.wav",
        "",
        H("🎙️ Voices"),
        f"  silero-onnx {F('--speaker-list')}",
        f"  silero-onnx {Q(_EX_07)} "
        f"{F('-o')} voice-baya.wav {F('--speaker')} baya",
        "",
        H("✍️ Stress marks (+) go BEFORE the stressed vowel and win"),
        f"  silero-onnx {Q(_EX_08)} "
        f"{F('-o')} castle.wav {F('--speaker')} baya",
        f"  silero-onnx {Q(_EX_09)} "
        f"{F('-o')} lock.wav {F('--speaker')} baya",
        "",
        H("⏱️ Punctuation shapes delivery (measured, not decorative: "
          "'...' resculpts phrasing, '—' adds a beat)"),
        f"  silero-onnx {Q(_EX_10)} "
        f"{F('-o')} dots.wav",
        f"  silero-onnx {Q(_EX_11)} "
        f"{F('-o')} dash.wav",
        "",
        H("⚡ SSML (needs --ssml; <speak> root is mandatory)"),
        f"  {G('<break> — exact pauses (ms, whole seconds, strength):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_12)} "
        f"{F('-o')} pause-500ms.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_13)} "
        f"{F('-o')} pause-2s.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_14)} "
        f"{F('-o')} pause-weak.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_15)} "
        f"{F('-o')} pause-x-strong.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_16)} "
        f"{F('-o')} pause-mixed.wav",
        f"  {G('<prosody> — the speech announces itself:')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_17)} "
        f"{F('-o')} rate.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_18)} "
        f"{F('-o')} pitch.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_19)} "
        f"{F('-o')} slow-low.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_20)} "
        f"{F('-o')} nested.wav",
        f"  {G('<p> and <s> — paragraph/sentence breaks (x-strong/strong):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_21)} "
        f"{F('-o')} sentences.wav",
        W("  NOTE: without --ssml, tags are plain text — <break .../> gets "
          "vocalized (numbers inside expand), it never pauses."),
        W("  NOTE: <intensity> and **/ * markers are v5_6_ru only."),
        "",
        H("📁 Custom models"),
        f"  silero-onnx {F('--models-dir')} D:\\models\\silero\\onnx "
        f"{Q(_EX_22)} {F('-o')} models-dir.wav",
    ])


def main(argv=None):
    # Examples contain Cyrillic: force UTF-8 so --help never dies on a
    # cp1252 console (same hardening as scripts/_common.py).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    del _s
    ap = argparse.ArgumentParser(
        prog="silero-onnx",
        usage="%(prog)s [options] [text]",
        description="Torch-free Silero RU TTS (48 kHz)",
        epilog="run `silero-onnx --examples` for usage examples",
        formatter_class=ColorHelpFormatter)
    ap.add_argument("text", nargs="?", help="Russian text to synthesize")
    ap.add_argument("-o", "--output", default="out.wav", help="output WAV path")
    ap.add_argument("--speaker", metavar="SPEAKER_NAME", default="xenia",
                    help="voice name (see --speaker-list)")
    ap.add_argument("--models-dir", metavar="MODELS_DIR", default=None,
                    help="dir with the self-contained .onnx model files")
    ap.add_argument("--model-version", metavar="VERSION",
                    default=DEFAULT_VERSION, choices=sorted(VERSIONS),
                    help=f"model set version: v5_6_ru (emphasis markers; "
                         f"default {DEFAULT_VERSION}) or v5_ru (classic "
                         f"pipeline)")
    ap.add_argument("--speaker-list", action="store_true",
                    help="list voices and exit (no models needed)")
    ap.add_argument("--model-list", action="store_true",
                    help="list model versions and exit (no models needed)")
    ap.add_argument("--ssml", action="store_true",
                    help="parse text as SSML (<speak>, break/prosody/p/s)")
    ap.add_argument("--examples", action="store_true",
                    help="print usage examples and exit")
    a = ap.parse_args(argv)
    if a.examples:
        print(examples_text())
        return
    if a.model_list:
        print(" ".join(sorted(VERSIONS)))
        return
    if a.speaker_list:
        try:
            table = load_speaker_map()
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(
                f"silero-onnx: error: cannot load speaker map: {e}")
        print(" ".join(sorted(table)))
        return
    model_dir = a.models_dir or _cli_models_dir(a.model_version)
    tts = SileroTTS(model_dir=model_dir, version=a.model_version)
    if not a.text:
        raise SystemExit("silero-onnx: error: no text given (see --help)")
    try:
        tts.synthesize_to_file(a.text, a.output, a.speaker, ssml=a.ssml)
    except ValueError as e:
        raise SystemExit(f"silero-onnx: error: {e}")
    import soundfile as sf
    info = sf.info(a.output)
    print(_paint("32", f"saved {a.output}: {info.duration:.2f}s, {a.speaker}"))
    assert "torch" not in sys.modules, "torch leaked into torch-free runtime!"


if __name__ == "__main__":
    main()
    assert "torch" not in sys.modules, "torch leaked!"
