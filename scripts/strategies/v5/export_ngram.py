"""Export the ngram accentor core -> ngram_core.onnx (single file).

Source: silero_stress accentor EmbeddingBag + stress/yo MLP heads
(`load_accentor().accentor.model`). String featurization (char n-grams
1..4 of `<word>`, dict lookup, UNK=126521 fallback, cumulative offsets)
stays in pure Python (runtime: silero_onnx/ngram_accentor.py); only the
numeric core becomes ONNX.

Inputs : ind (K,) int64 ngram ids, offsets (W,) int64 row starts.
         K (total ngrams) and W (words) are dynamic.
Outputs: stress_logits (W,S), yo_logits (W,Y) fp32 (softmax+argmax in numpy).

NOTE (do not "fix"): EmbeddingBag is built with max_norm=None. An
earlier max_norm=2.0 assumption clipped rows and broke exactly the words
whose embedding norm exceeded 2 — verified regression.

Gates: L1 onnx.checker + ORT vs torch on 10 featurized words, atol=1e-4;
then L2 stochastic (5 seeded random id bags, atol=1e-3).

Usage (from the repo root, build env installed):
  pip install -r scripts/requirements-export.txt
  python scripts/strategies/v5/export_ngram.py [--out-dir onnx]

Result: <out-dir>/ngram_core.onnx.
"""
import argparse
import os
import sys

# Path bootstrap for `python scripts/strategies/v5/<this>.py`
# (v5 blocks live next to this file; shared _common in scripts/).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(os.path.dirname(_HERE)), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _HERE, _p
from _common import (DEFAULT_OUT_DIR, Logger, begin, check_and_gate,
                     collapse_call, finish, seed_all, stochastic_probe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="destination dir for ngram_core.onnx")
    a = ap.parse_args()

    import torch

    from silero_stress import load_accentor

    log = Logger()
    begin(log, "export_ngram.py", "ngram_core.onnx")
    os.makedirs(a.out_dir, exist_ok=True)
    onnx_path = os.path.join(a.out_dir, "ngram_core.onnx")

    acc = load_accentor()

    def get(root, *names):
        m = root
        for n in names:
            m = dict([(k, v) for k, v in m.named_children()])[n]
        return m

    core = acc.accentor.model
    ngram_dict = get(core, "embedding").ngram_dict
    WORDS = ["замок", "был", "открыт", "меня", "зовут", "лева",
             "королев", "и", "уже", "готов"]

    def featurize(words):
        unk_id = 126521
        ids, offs, total = [], [0], 0
        for w in words:
            s = "<" + w + ">"
            hit = []
            for n in (1, 2, 3, 4):
                for i in range(len(s) - n + 1):
                    g = s[i:i + n]
                    if g in ngram_dict:
                        hit.append(ngram_dict[g])
            if not hit:
                hit = [unk_id]
            ids += hit
            total += len(hit)
            offs.append(total)
        return torch.LongTensor(ids), torch.LongTensor(offs[:-1])

    ids, offs = featurize(WORDS)
    log.log(f"featurized: {len(ids)} ids, offsets {offs.tolist()}")

    class NgramCore(torch.nn.Module):
        def __init__(self, src):
            super().__init__()
            emb_src = get(src, "embedding")
            self.emb = torch.nn.EmbeddingBag(len(emb_src.weight), 16,
                                             mode="mean")
            with torch.no_grad():
                self.emb.weight.copy_(emb_src.weight)
            self.stress_clf = get(src, "stress_clf")
            self.yo_clf = get(src, "yo_clf")

        def forward(self, ind, offsets):
            h = self.emb(ind, offsets)
            return self.stress_clf(h), self.yo_clf(h)

    wrap = NgramCore(core).eval()
    with torch.no_grad():
        ref = wrap(ids, offs)
    log.log(f"ref shapes: {[tuple(t.shape) for t in ref]}")

    traced = torch.jit.trace(wrap, (ids, offs), strict=False)
    collapse_call(torch.onnx.export,
        traced, (ids, offs), onnx_path,
        input_names=["ind", "offsets"],
        output_names=["stress_logits", "yo_logits"],
        dynamic_axes={"ind": {0: "K"}, "offsets": {0: "W"},
                      "stress_logits": {0: "W"}, "yo_logits": {0: "W"}},
        opset_version=17, dynamo=False)
    log.log("export=OK")

    _, ok = check_and_gate(onnx_path,
                           {"ind": ids.numpy(), "offsets": offs.numpy()},
                           [ref[0].numpy(), ref[1].numpy()],
                           ["stress", "yo"], log.log, atol=1e-4)

    # Level-2 (stochastic robustness): seeded random id bags with valid
    # cumulative offsets, varying word counts. Relaxed (1e-3).
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR only (see engine.py)
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    seed_all()
    n_rows = int(get(core, "embedding").weight.shape[0])
    l2_cases = []
    with torch.no_grad():
        for W_ in (1, 3, 7, 2, 5):
            lens = torch.randint(1, 30, (W_,))
            coffs = torch.zeros(W_, dtype=torch.long)
            coffs[1:] = torch.cumsum(lens[:-1], 0)
            cind = torch.randint(0, n_rows, (int(lens.sum()),))
            rs, ry = wrap(cind, coffs)
            l2_cases.append(([rs.numpy(), ry.numpy()],
                             {"ind": cind.numpy(),
                              "offsets": coffs.numpy()}))
    ok &= stochastic_probe(sess, l2_cases, ["stress", "yo"], log.log,
                           atol=1e-3)
    finish(log, os.path.join(a.out_dir, "ngram_export_result.txt"),
           [onnx_path], "export_ngram", ok)


if __name__ == "__main__":
    main()
