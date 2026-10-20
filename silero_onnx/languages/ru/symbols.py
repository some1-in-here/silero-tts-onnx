"""Russian symbol/speaker tables and tokenization helpers."""
import json
import os

from ..base import add_sos_eos, tokenize

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data")


def load_symbol_table(data_dir=DATA_DIR):
    """Torch-free symbol table (dumped + validated by the build)."""
    with open(os.path.join(data_dir, "symbols.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["symbols"], d["symbol_to_id"]


def load_speaker_map(data_dir=DATA_DIR):
    """Torch-free speaker name -> embedding id (dumped once, verified)."""
    with open(os.path.join(data_dir, "speakers.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {k: v["id"][0] for k, v in d.items()}


__all__ = ["DATA_DIR", "load_symbol_table", "load_speaker_map",
           "tokenize", "add_sos_eos"]
