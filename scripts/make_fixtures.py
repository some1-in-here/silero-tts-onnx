"""Generate build-time reference fixtures (deterministic, seed 42).

Writes <out-dir>/fixtures/ (git-ignored build output, NOT versioned):
  - stress_ref.pt: {ids, starts, ends} — 7 real homograph windows.
    Built with the vendored tokenizer + homodict span tagging (the same
    code the runtime uses), no torch model needed.
  - spec_ref.pt: {mag, x, y} — real spectrum from the clean acoustic +
    torch vocoder on a fixed text (needs the torch.hub model).

Both are *inputs* to the export gates (torch-vs-torch fidelity and
ORT-vs-torch references), never to tests: the suite is torch-free and
must run without them.

make_spec() additionally validates the versioned runtime tables in
silero_onnx/data/ (speakers.json, symbols.json) against the hub
package — write-or-fail, never silent (see _common.write_*_json).

Runs automatically as step 0 of export_all.py, or standalone:

  python scripts/make_fixtures.py [--out-dir onnx]
"""
import argparse
import os
import sys

# v5 build blocks moved to scripts/strategies/v5/ (symmetric layout).
_V5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "strategies", "v5")
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)
del _V5_DIR

from _common import (DEFAULT_OUT_DIR, Logger, begin, finish, fixtures_dir,
                     package_data_path, write_speakers_json,
                     write_symbols_json)


# Fixed sentence pool: must yield >= 7 neural (non-regex) homograph
# windows through OnnxStress tagging. Order is load-bearing for byte
# determinism — do not reorder, only append.
SENTENCES = [
    "На горе стоит старый замок.",
    "Мел рассыпался по полу.",
    "Мука уже закончилась.",
    "Хлопок раздался внезапно.",
    "Ирис растет в саду.",
    "Замок был открыт.",
    "Полки прогнулись под книгами.",
    "Кружки стоят на столе.",
    "Сорока сидела на ветке.",
    "Он ел потом и кровью.",
    "Белки прыгали по веткам.",
    "Дорога шла через поле.",
    "Гвозди торчали из доски.",
    "Стрелки часов замерли.",
]

SPEC_TEXT = "Привет! Меня зовут Ксения."
N_WINDOWS = 7


def make_stress(log):
    """Tokenize real sentences, collect the first 7 neural homo windows."""
    import numpy as np

    from silero_onnx.homograph_solver import OnnxStress

    # NOTE: no seed_all() here — it imports torch, and importing
    # silero_onnx.* afterwards trips the torch-free guard in engine.py.
    # Tagging is fully deterministic anyway (fixed strings + tables).
    st = OnnxStress(models_dir=None)  # session stays lazy; never touched
    b_starts, b_ends, b_sents = [], [], []
    for sent in SENTENCES:
        for _s, _e, _w, wl, raw_mark, _c in st._find_and_tag_homos(sent):
            if raw_mark is None or wl not in st.homodict:
                continue
            ids = np.array(st.tokenizer(raw_mark), dtype=np.int64)
            hs = np.where(ids == st.tokenizer.homo_start_id)[0]
            he = np.where(ids == st.tokenizer.homo_end_id)[0]
            if len(hs) == 0 or len(he) == 0:
                continue
            b_starts.append(int(hs[0]))
            b_ends.append(int(he[0]))
            b_sents.append(ids)
            if len(b_sents) == N_WINDOWS:
                break
        if len(b_sents) == N_WINDOWS:
            break
    if len(b_sents) < N_WINDOWS:
        raise SystemExit(
            f"make_fixtures: only {len(b_sents)}/{N_WINDOWS} homo windows "
            f"found — extend SENTENCES")
    width = max(len(s) for s in b_sents)
    pad = st.tokenizer.pad_token_id
    batch = np.stack([np.pad(s, (0, width - len(s)), constant_values=pad)
                      for s in b_sents]).astype(np.int64)
    log.log(f"stress: {N_WINDOWS} windows, T={width} "
            f"(starts={b_starts} ends={b_ends})")
    return {"ids": batch,
            "starts": np.array(b_starts, dtype=np.int64),
            "ends": np.array(b_ends, dtype=np.int64)}


def make_spec(log):
    """Real spectrum via clean acoustic + torch vocoder (fixed text)."""
    import torch

    from build_acoustic import build_acoustic

    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker="v5_ru", trust_repo=True)
    model.to("cpu")
    part = model.packages[0]
    jit_m = part.models[0]

    # Speaker map for the torch-free runtime: names AND ids come from
    # the hub package's own get_speakers() (never positional guesses).
    # Written to the versioned package data (review + commit the diff);
    # a later version with a different table fails loudly in
    # write_speakers_json instead of overwriting.
    entries = [(name, *part.get_speakers(name)) for name in part.speakers]
    entries = [(name, ids[0], mid) for name, ids, mid in entries]
    write_speakers_json(entries, package_data_path("speakers.json"),
                        log.log)

    # Same write-or-fail discipline for the shared symbol table: the
    # runtime package reads it but must never import torch to
    # regenerate it — that one hub read lives here, in the build.
    write_symbols_json(list(part.symbols), dict(part.symbol_to_id),
                       package_data_path("symbols.json"), log.log)

    def get(root, *names):
        m = root
        for n in names:
            m = dict(m.named_children())[n]
        return m

    new = build_acoustic(jit_m, [])
    voc = get(jit_m, "vocoder")
    spk_ids, _mid = part.get_speakers("xenia")
    sents, _, blens, rates, pitches, sp_ids = part.prepare_tts_model_input(
        SPEC_TEXT, ssml=False, speaker_ids=spk_ids)
    seq, _, dr, pc = part.merge_batch_model(sents, blens, rates, pitches,
                                            put_accent=True, put_yo=True)
    with torch.no_grad():
        mel, _dur = new(seq, sp_ids, dr.clone(), pc.clone())
        mag, x, y = voc(mel, 48000, 0.0, True)
    log.log(f"spec: {SPEC_TEXT!r} -> mag{tuple(mag.shape)}")
    return {"mag": mag, "x": x, "y": y, "mel": mel}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination root (fixtures go to <out-dir>/fixtures)")
    a = ap.parse_args()

    log = Logger()
    begin(log, "make_fixtures.py", "fixtures/stress_ref.pt + spec_ref.pt")
    fx = fixtures_dir(a.out_dir)
    os.makedirs(fx, exist_ok=True)

    # NOTE: stress FIRST and torch-free: importing silero_onnx.* after
    # torch trips the torch-free guard in engine.py.
    stress = make_stress(log)

    import torch

    torch.save({"ids": torch.from_numpy(stress["ids"]),
                "starts": torch.from_numpy(stress["starts"]),
                "ends": torch.from_numpy(stress["ends"])},
               os.path.join(fx, "stress_ref.pt"))
    log.log("wrote stress_ref.pt")

    spec = make_spec(log)
    spec_path = os.path.join(fx, "spec_ref.pt")
    torch.save(spec, spec_path)
    finish(log, os.path.join(a.out_dir, "fixtures_result.txt"),
           [os.path.join(fx, "stress_ref.pt"), spec_path],
           "make_fixtures", True)


if __name__ == "__main__":
    main()
