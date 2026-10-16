"""v5_6_ru investigation probe: symbols, structure, emphasis behavior.

- Loads v5_ru (baseline) and v5_6_ru via torch.hub (no force_reload:
  v5_ru.pt may be file-locked; plain reads are lock-free).
- Diffs symbol tables and speaker lists.
- Summarizes module dims (validates transplant compatibility).
- Compares audio on plain vs emphasis-marked inputs (CAPS, *stars*,
  '!', '?') within and across versions.
Kept (not throwaway): documents how v5_6 was measured for the export.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import torch

torch._C._jit_set_profiling_mode(False)


def load(speaker):
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker=speaker, trust_repo=True)
    model.to("cpu")
    return model


def get(root, *names):
    m = root
    for n in names:
        m = dict(m.named_children())[n]
    return m


print("== loading v5_ru / v5_6_ru ==", flush=True)
m5, m6 = load("v5_ru"), load("v5_6_ru")
p5, p6 = m5.packages[0], m6.packages[0]

s5, s6 = list(p5.symbols), list(p6.symbols)
print(f"symbols: v5_ru={len(s5)} v5_6_ru={len(s6)}", flush=True)
print("only in v5_ru :", sorted(set(s5) - set(s6)), flush=True)
print("only in v5_6_ru:", sorted(set(s6) - set(s5)), flush=True)
print("speakers v5_6_ru:", p6.speakers, flush=True)

for tag, part in (("v5_ru", p5), ("v5_6 ", p6)):
    jm = part.models[0]
    taco = get(jm, "tacotron")
    enc = get(taco, "encoder")
    print(f"{tag}: symbols={get(taco, 'embedding').weight.shape[0]} "
          f"d_model={get(taco, 'embedding').weight.shape[1]} "
          f"enc_layers={len(list(enc.layers.children()))} "
          f"speakers={get(taco, 'speaker_embedding').weight.shape[0]} "
          f"mel={get(taco, 'lin').weight.shape[0]} "
          f"energy_proj={'energy_proj' in dict(taco.named_children())}",
          flush=True)
    for name in ("dur_predictor", "pitch_predictor"):
        inner = get(jm, name,
                    "dur_pred" if "dur" in name else "pitch_pred")
        rows = {n: (tuple(m.weight.shape) if hasattr(m, "weight") else None)
                for n, m in inner.named_children()
                if "embedding" in n}
        print(f"{tag} {name}: {rows}", flush=True)

CASES = {
    "plain": "Привет! Меня зовут Ксения.",
    "caps": "ПРИВЕТ! МЕНЯ ЗОВУТ КСЕНИЯ.",
    "stars": "Привет! Меня зовут *Ксения*.",
    "qmark": "Меня зовут Ксения?",
    "excl": "Меня зовут Ксения!",
}
with torch.no_grad():
    for name, text in CASES.items():
        a5 = np.asarray(p5.apply_tts(text=text, speaker="xenia",
                                     sample_rate=48000)).ravel()
        a6 = np.asarray(p6.apply_tts(text=text, speaker="xenia",
                                     sample_rate=48000)).ravel()
        n = min(len(a5), len(a6))
        d = float(np.abs(a5[:n] - a6[:n]).max())
        print(f"{name}: v5_ru {len(a5)} vs v5_6 {len(a6)} "
              f"samples, cross max_abs={d:.3e}", flush=True)
