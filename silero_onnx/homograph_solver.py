"""Torch-free stress inference: vendored tokenizer + data + ORT model.

Replicates HomoSolver.__call__ (homosolver.py lines 44-139) with numpy:
tag/clean/regex/phrase logic is plain Python; only the neural batch goes
to silero_stress.onnx. No torch import anywhere in this file.
Data files (versioned runtime tables, see silero_onnx/data/):
silero_onnx/data/{homodict,phrases,tokenizer}.json, plus a tokdir/
rebuilt at runtime under the user cache (NOT silero_onnx/data/tokdir/,
which is an untracked leftover) for the vendored tokenizer.
"""
import json
import os
import re

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _cache_dir():
    """Writable per-user cache (package dir may be read-only when installed)."""
    from . import env as _env

    base = _env.cache_dir() or os.path.join(
        os.path.expanduser("~"), ".cache", "silero_onnx")
    os.makedirs(base, exist_ok=True)
    return base


def _ensure_tokdir():
    """Reconstruct the tokenizer_dir layout the vendored tokenizer expects.

    Generated under the user cache (NOT the package dir): the installed
    wheel may live in a read-only site-packages. Content is deterministic
    (rebuilt from tokenizer.json); a marker file guards completeness.
    """
    tokdir = os.path.join(_cache_dir(), "tokdir")
    marker = os.path.join(tokdir, ".ok")
    if os.path.isfile(marker) and os.path.isfile(os.path.join(tokdir, "vocab.txt")):
        return tokdir
    os.makedirs(tokdir, exist_ok=True)
    tj = json.load(open(os.path.join(DATA, "tokenizer.json"), encoding="utf-8"))
    cfg, vocab = tj["config"], tj["vocab"]
    with open(os.path.join(tokdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"never_split": cfg["never_split"],
                   "tokenize_chinese_chars": cfg["tokenize_chinese_chars"],
                   "model_max_length": cfg["model_max_length"]}, f)
    with open(os.path.join(tokdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({"unk_token": cfg["unk_token"], "sep_token": cfg["sep_token"],
                   "pad_token": cfg["pad_token"], "cls_token": cfg["cls_token"],
                   "mask_token": cfg["mask_token"]}, f)
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in ordered] == list(range(len(ordered))), "vocab ids must be dense"
    with open(os.path.join(tokdir, "vocab.txt"), "w", encoding="utf-8") as f:
        for tok, _ in ordered:
            f.write(tok + "\n")
    with open(os.path.join(tokdir, "added_tokens.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("silero-onnx tokdir v1\n")
    return tokdir


def _capitalize_stress(w):
    i = w.index("+")
    return w[:i] + w[i + 1].capitalize() + w[i + 2:]


def _decapitalize_stress(w):
    for idx, ch in enumerate(w):
        if ch.isupper():
            return w[:idx] + "+" + w[idx:].lower()
    return w


class OnnxStress:
    """Drop-in replacement for pip accentor __call__ minus torch."""

    def __init__(self, models_dir=None, data_dir=DATA):
        from .bert_tokenizer import SimpleBertTokenizer
        self.tokenizer = SimpleBertTokenizer(tokenizer_dir=_ensure_tokdir())
        with open(os.path.join(data_dir, "homodict.json"), encoding="utf-8") as f:
            self.homodict = json.load(f)
        self.yohomodict = {w: v for w, v in self.homodict.items()
                           if any("ё" in x for x in v)}
        with open(os.path.join(data_dir, "phrases.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.compiled_phrases = {w: re.compile(v["pattern"], v["flags"])
                                 for w, v in raw.items()}
        self.compiled_user_phrases = {}
        self.compiled_user_patterns = {}
        with open(os.path.join(data_dir, "stress_cfg.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.vowels = cfg["vowels"]
        self.window_size = cfg["window_size"]
        self.pattern = re.compile(cfg["tag_pattern"], cfg["tag_flags"])
        self._re_remove_extra = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.!?,\-]')
        self._re_spaces = re.compile(r'\s+')
        self._re_double_dash = re.compile(r'-{2,}')
        self._re_repeat_punct = re.compile(r'([.!?])\1+')
        self._re_repeat_comma = re.compile(r',{2,}')
        self._re_space_before_punct = re.compile(r'\s+([.,!?])')
        self._re_punct_add_space = re.compile(r'([.,!?])(?=\S)')
        self.device = "cpu"
        self._sess = None
        self._models_dir = models_dir

    def _session(self):
        if self._sess is None:
            from . import env as _env
            from .ort import create_session

            if self._models_dir:
                path = os.path.join(self._models_dir, "silero_stress.onnx")
            else:
                envd = _env.models_dir()
                path = os.path.join(envd, "silero_stress.onnx") if envd else None
            if not path or not os.path.isfile(path):
                raise SystemExit(
                    "silero-onnx: error: missing model file " + str(path) +
                    " (pass models_dir explicitly or set SILERO_ONNX_MODELS)")
            self._sess = create_session(path)
        return self._sess

    # -- text plumbing, ported 1:1 from homosolver.py --
    def _clean_text(self, text, is_start=True):
        if not text:
            return ""
        text = self._re_remove_extra.sub('', text)
        text = self._re_spaces.sub(' ', text)
        text = self._re_double_dash.sub(' - ', text)
        text = self._re_repeat_punct.sub(r'\1', text)
        text = self._re_repeat_comma.sub(',', text)
        text = self._re_space_before_punct.sub(r'\1', text)
        text = self._re_repeat_punct.sub(r'\1', text)
        text = self._re_repeat_comma.sub(',', text)
        text = self._re_punct_add_space.sub(r'\1 ', text)
        text = self._re_spaces.sub(' ', text).strip()
        if is_start:
            text = text.lstrip(' .,!?-')
            if text:
                text = text[0].upper() + text[1:].lower()
        else:
            if text:
                text = text[0] + text[1:].lower()
            if text and text[-1] not in '.!?':
                text += '.'
        return text

    def _find_and_tag_homos(self, sentence, words_to_ignore=None):
        words_to_ignore = words_to_ignore if words_to_ignore is not None else []
        out = []
        for match in self.pattern.finditer(sentence):
            start, end = match.span()
            word = match.group()
            wl = word.lower()
            is_h = ((wl in self.homodict) or (wl in self.compiled_phrases)
                    or (wl in self.compiled_user_phrases)
                    or (wl in self.compiled_user_patterns)) and (wl not in words_to_ignore)
            if is_h:
                pre = self._clean_text(sentence[:start], is_start=True)[(-self.window_size // 2):]
                post = self._clean_text(sentence[end:], is_start=False)[:(self.window_size // 2)]
                raw_mark = (pre + ' [HOMO] ' + wl + ' [/HOMO] ' + post).strip()
                raw_clean = (pre + ' ' + wl + ' ' + post).strip()
            else:
                raw_mark = raw_clean = None
            out.append((start, end, word, wl, raw_mark, raw_clean))
        return out

    def _predict_with_pattern(self, pattern, text):
        match = pattern.search(text)
        if match:
            for group_name, matched_text in match.groupdict().items():
                if matched_text is not None:
                    return _decapitalize_stress(group_name)
        return None

    def __call__(self, sentence, put_stress=True, put_yo=True,
                 stress_single_vowel=True, words_to_ignore=None,
                 put_stress_homo=True, put_yo_homo=True):
        _ = (put_stress_homo, put_yo_homo)  # accepted for API parity
        if not (put_stress or put_yo):
            return sentence
        tagged = self._find_and_tag_homos(sentence, words_to_ignore=words_to_ignore)
        b_starts, b_ends, b_sents = [], [], []
        starts, ends, words, regex_preds, is_neural = [], [], [], [], []
        for start, end, word, wl, raw_mark, raw_clean in tagged:
            if raw_mark is None:
                continue
            regex_pred = None
            if wl in self.compiled_user_patterns:
                regex_pred = self._predict_with_pattern(
                    self.compiled_user_patterns[wl], raw_clean)
            if regex_pred is None and wl in self.compiled_user_phrases:
                regex_pred = self._predict_with_pattern(
                    self.compiled_user_phrases[wl], raw_mark)
            if regex_pred is None and wl in self.compiled_phrases:
                regex_pred = self._predict_with_pattern(
                    self.compiled_phrases[wl], raw_mark)
            if regex_pred is not None:
                regex_preds.append(regex_pred)
            else:
                if wl not in self.homodict:
                    continue
                ids = np.array(self.tokenizer(raw_mark), dtype=np.int64)
                hs_ = np.where(ids == self.tokenizer.homo_start_id)[0][0]
                he_ = np.where(ids == self.tokenizer.homo_end_id)[0][0]
                b_starts.append(hs_)
                b_ends.append(he_)
                b_sents.append(ids)
            starts.append(start)
            ends.append(end)
            words.append(word)
            is_neural.append(regex_pred is None)
        if len(words) == 0:
            return sentence
        homosolver_preds = None
        if len(b_sents) > 0:
            width = max(len(s) for s in b_sents)
            pad = self.tokenizer.pad_token_id
            batch = np.stack([np.pad(s, (0, width - len(s)), constant_values=pad)
                              for s in b_sents]).astype(np.int64)
            logits = self._session().run(
                None, {"input_ids": batch,
                       "homo_start_ids": np.array(b_starts, dtype=np.int64),
                       "homo_end_ids": np.array(b_ends, dtype=np.int64)})[0]
            homosolver_preds = (1 / (1 + np.exp(-logits))).round().astype(int)
        regex_idx = homosolver_idx = 0
        word_preds = []
        for word, neural in zip(words, is_neural):
            if neural:
                pred = int(homosolver_preds[homosolver_idx][0])
                word_preds.append(sorted(self.homodict[word.lower()])[pred])
                homosolver_idx += 1
            else:
                word_preds.append(regex_preds[regex_idx])
                regex_idx += 1
        stressed, offset = sentence, 0
        for start, end, word, wp in zip(starts, ends, words, word_preds):
            start, end = start + offset, end + offset
            wp = wp if put_yo else wp.replace('ё', 'е')
            n_vowels = sum(c.lower() in self.vowels for c in wp)
            stress_idx = wp.index('+')
            wp = wp.replace('+', '')
            wp = ''.join([c2.lower() if c1.islower() else c2.upper()
                          for c1, c2 in zip(word, wp)])
            if (n_vowels > 1 or stress_single_vowel) and put_stress:
                wp = wp[:stress_idx] + '+' + wp[stress_idx:]
                offset += 1
            stressed = stressed[:start] + wp + stressed[end:]
        return stressed


class FullStress:
    """Torch-free SileroStress: homosolver-ONNX then ngram-ONNX.

    Replicates models/model.py SileroStress.__call__ exactly, including
    the skip-list wiring (homodict/yohomodict passed as skip lists when
    the corresponding homo flags are off).
    """

    def __init__(self, models_dir=None, data_dir=DATA):
        self.homo = OnnxStress(models_dir=models_dir, data_dir=data_dir)
        from .ngram_accentor import OnnxNgram
        self.ngram = OnnxNgram(models_dir=models_dir, data_dir=data_dir)

    def __call__(self, sentence, put_stress=True, put_stress_homo=True,
                 put_yo=True, put_yo_homo=True, stress_single_vowel=True,
                 words_to_ignore=None):
        if words_to_ignore is not None and not isinstance(
                words_to_ignore, (list, set, tuple)):
            raise ValueError("`words_to_ignore` should be either None or list/set/tuple")
        mid = self.homo(sentence, put_stress=put_stress_homo, put_yo=put_yo_homo,
                        stress_single_vowel=stress_single_vowel,
                        words_to_ignore=words_to_ignore)
        return self.ngram(
            mid, put_stress=put_stress, put_yo=put_yo,
            stress_single_vowel=stress_single_vowel,
            skip_stress_words=(self.homo.homodict if not put_stress_homo else None),
            skip_yo_words=(self.homo.yohomodict if not put_yo_homo else None),
            words_to_ignore=words_to_ignore)
