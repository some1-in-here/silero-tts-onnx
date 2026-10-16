"""CLI + model discovery for torch-free Silero TTS (text -> ONNX -> audio).

Inference itself lives in strategies/ (see strategies/__init__.py);
this module keeps argument parsing, console output, and the
model-directory discovery. Nothing here imports torch (verified:
fresh subprocess + ``sys.modules`` check in CI-style tests).

Typical use:
  tts = SileroTTS(model_dir="path/to/onnx", version="v5_ru")
  audio = tts.synthesize("Привет, мир!", speaker="xenia")  # float32 mono, 48kHz
"""
import argparse
import os
import sys

import numpy as np  # noqa: F401 (kept for console-side array handling)

from . import env
from .frontend import load_speaker_map
from .strategies import SileroTTS
from .versions import DEFAULT_VERSION, MODEL_SETS, VERSIONS

assert "torch" not in sys.modules, "torch must never be imported here"


def _env_models_dir(version=DEFAULT_VERSION):
    """SILERO_ONNX_MODELS env var, validated by the version's acoustic file.

    Explicit user configuration: honored by both the library and the
    CLI. Returns None when unset; a set-but-wrong path fails loudly
    downstream (missing-file diagnostics), never silently falls back.
    """
    acoustic = MODEL_SETS.get(version, {}).get("acoustic", "")
    envdir = env.models_dir()
    if envdir and acoustic and os.path.isfile(os.path.join(envdir, acoustic)):
        return envdir
    return None


def _cli_models_dir(version=DEFAULT_VERSION):
    """Model directory autoresolve — CLI ONLY (env var, then <repo>/models/onnx).

    The `<repo>/models/onnx` fallback lives here and nowhere else: library
    entry points (`SileroTTS`, `create_strategy`) never look there and
    fail loudly without an explicit `model_dir` (or the env var).
    """
    found = _env_models_dir(version)
    if found is not None:
        return found
    acoustic = MODEL_SETS.get(version, {}).get("acoustic", "")
    here = os.path.dirname(os.path.abspath(__file__))
    repo_onnx = os.path.normpath(
        os.path.join(here, "..", "models", "onnx"))
    if acoustic and os.path.isfile(os.path.join(repo_onnx, acoustic)):
        return repo_onnx
    return None


def _dev_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _use_color():
    """ANSI colors only for live terminals that allow them."""
    return env.colors_enabled()


def _paint(code, s):
    return f"\033[{code}m{s}\033[0m" if _use_color() else s


class ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """argparse help with section/flag coloring (tty only, via _paint).

    --speakers output stays plain on purpose (machine-readable list).
    """

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
# backslash escapes) so the module parses on Python 3.10+.
_EX_00 = '"Привет! Меня зовут Ксения."'
_EX_01 = '"Замок был открыт."'
_EX_02 = '"З+амок был открыт."'
_EX_03 = '"Зам+ок был открыт."'
_EX_04 = '"Подожди... Продолжаю."'
_EX_05 = '"Он — молодец."'
_EX_06 = '"<speak>Сейчас будет пауза. <break time=\'500ms\'/> Вот она была.</speak>"'
_EX_07 = '"<speak>Сейчас будет пауза. <break time=\'2s\'/> Вот она была.</speak>"'
_EX_08 = '"<speak>Сейчас будет пауза. <break strength=\'weak\'/> Вот она была.</speak>"'
_EX_09 = '"<speak>Первая часть. <break strength=\'x-strong\'/> Вторая часть после очень длинной паузы.</speak>"'
_EX_10 = '"<speak>Раз. <break time=\'250ms\'/> Два. <break strength=\'strong\'/> Три.</speak>"'
_EX_11 = '"<speak><prosody rate=\'fast\'>Я говорю быстро-быстро!</prosody> <prosody rate=\'slow\'>А теперь медленно и размеренно.</prosody>А теперь как обычно.</speak>"'
_EX_12 = '"<speak><prosody pitch=\'high\'>Здесь я говорю высоко.</prosody> <prosody pitch=\'low\'>А здесь низко.</prosody></speak>"'
_EX_13 = '"<speak><prosody rate=\'slow\' pitch=\'low\'>Говорю медленно и низко.</prosody></speak>"'
_EX_14 = '"<speak><prosody rate=\'slow\'>Говорю медленно, <prosody pitch=\'high\'>а это слово высоко</prosody> и снова ровно.</prosody></speak>"'
_EX_15 = '"<speak><s>Первое.</s><s>Второе.</s></speak>"'
_EX_16 = '"Текст."'
_EX_17 = '"Меня зовут **Ксения**."'
_EX_18 = '"Меня зовут *Ксения*."'
_EX_19 = '"<intensity=1>Тихо говорю.</intensity>"'


def examples_text():
    """Structured, color-aware usage examples (see --examples)."""
    c = _use_color()

    def H(s):  # yellow: section headers
        return f"\033[33m{s}\033[0m" if c else s

    def F(s):  # green: flags
        return f"\033[32m{s}\033[0m" if c else s

    def Q(s):  # cyan: quoted text
        return f"\033[36m{s}\033[0m" if c else s

    def G(s):  # gray: comments
        return f"\033[90m{s}\033[0m" if c else s

    def W(s):  # red: warnings
        return f"\033[31m{s}\033[0m" if c else s

    return "\n".join([
        H("📌 Basic"),
        f"  silero-onnx {Q(_EX_00)} "
        f"{F('-o')} out.wav",
        "",
        H("🎯 Versions (default is v5_6_ru)"),
        f"  silero-onnx {Q(_EX_00)} {F('-o')} out.wav "
        f"{G('# v5_6_ru: emphasis markers, no SSML')}",
        f"  silero-onnx {F('--model-version')} v5_ru {Q(_EX_00)} "
        f"{F('-o')} out5.wav {G('# classic pipeline: SSML works here')}",
        f"  silero-onnx {F('--model-list')} "
        f"{G('# show available versions (needs no models)')}",
        "",
        H("✨ Emphasis (v5_6_ru only: **, *, <intensity>)"),
        f"  silero-onnx {Q(_EX_17)} {F('-o')} emph.wav "
        f"{G('# **word**: emphasized')}",
        f"  silero-onnx {Q(_EX_18)} {F('-o')} focus.wav "
        f"{G('# *word*: focused')}",
        f"  silero-onnx {Q(_EX_19)} {F('-o')} quiet.wav "
        f"{G('# <intensity=0..3>: focus strength, default 3')}",
        "",
        H("🎙️ Voices"),
        f"  silero-onnx {F('--speaker-list')} "
        f"{G('# show voices (needs no models)')}",
        f"  silero-onnx {Q(_EX_01)} {F('-o')} castle.wav "
        f"{F('--speaker')} baya",
        "",
        H("✍️ Stress marks (+) go BEFORE the stressed vowel and win"),
        f"  {G('# castle vs lock: з+амок / зам+ок (system convention, see homodict)')}",
        f"  silero-onnx {Q(_EX_02)} {F('-o')} castle1.wav "
        f"{F('--speaker')} baya",
        f"  silero-onnx {Q(_EX_03)} {F('-o')} castle2.wav "
        f"{F('--speaker')} baya",
        "",
        H("⏱️ Punctuation shapes delivery (measured, not decorative)"),
        f"  silero-onnx {Q(_EX_04)} {F('-o')} dots.wav "
        f"{G('# ... resculpts phrasing')}",
        f"  silero-onnx {Q(_EX_05)} {F('-o')} dash.wav "
        f"{G('# em-dash adds a beat')}",
        "",
        H("⚡ SSML (v5_ru only; needs --ssml; <speak> root is mandatory)"),
        f"  {G('<break> — exact pauses (ms, whole seconds, strength):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_06)} "
        f"{F('-o')} br.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_07)} "
        f"{F('-o')} br2.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_08)} "
        f"{F('-o')} br3.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_09)} "
        f"{F('-o')} br4.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_10)} "
        f"{F('-o')} br5.wav",
        f"  {G('<prosody> — the speech announces itself:')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_11)} "
        f"{F('-o')} pr.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_12)} "
        f"{F('-o')} pi.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_13)} "
        f"{F('-o')} pi2.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_14)} "
        f"{F('-o')} pi3.wav {G('# nested prosody')}",
        f"  {G('<p> and <s> — paragraph/sentence breaks (x-strong/strong):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q(_EX_15)} "
        f"{F('-o')} ps.wav",
        W("  NOTE: without --ssml, tags are plain text — <break .../> gets "
          "vocalized (numbers inside expand), it never pauses."),
        "",
        H("📁 Custom models"),
        f"  silero-onnx {F('--models-dir')} D:\\models\\silero\\onnx "
        f"{Q(_EX_16)} {F('-o')} out.wav",
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
                      help=f"model set version: v5_ru (SSML-capable classic) "
                           f"or v5_6_ru (emphasis markers; default "
                           f"{DEFAULT_VERSION})")
    ap.add_argument("--speaker-list", action="store_true",
                      help="list voices and exit (no models needed)")
    ap.add_argument("--model-list", action="store_true",
                      help="list model versions and exit (no models needed)")
    ap.add_argument("--ssml", action="store_true",
                      help="parse text as SSML (<speak>, break/prosody/p/s; "
                           "v5_ru only)")
    ap.add_argument("--examples", action="store_true",
                      help="print usage examples and exit")
    a = ap.parse_args(argv)
    # Light commands first: they need no models and must work on a bare
    # checkout (previously --speakers built SileroTTS first, so it failed
    # without models). Voices come from the versioned speaker set, which
    # is identical for every version.
    if a.examples:
        print(examples_text())
        return
    if a.model_list:
        print(" ".join(sorted(VERSIONS)))
        return
    if a.speaker_list:
        # No models, no constants: straight from the shipped speakers.json,
        # so a regenerated table (new upstream voice) lists automatically.
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
        # Expected input failures (bad SSML, unknown speaker/symbols):
        # clean one-line message, never a traceback.
        raise SystemExit(f"silero-onnx: error: {e}")
    import soundfile as sf
    info = sf.info(a.output)
    print(_paint("32", f"saved {a.output}: {info.duration:.2f}s, {a.speaker}"))
    assert "torch" not in sys.modules, "torch leaked into torch-free runtime!"


if __name__ == "__main__":
    main()
    assert "torch" not in sys.modules, "torch leaked!"
