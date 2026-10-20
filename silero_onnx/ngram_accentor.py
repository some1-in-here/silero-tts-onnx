"""Torch-free Ngram accentor: featurization + ORT core + rules.

Faithful port of models/accentor.py (AccentorNgram) with numpy/onnxruntime:
- word_ngrams/min/max + dict lookup + UNK fallback + offsets (from the
  TorchScript embedding graph, modules.py:108-133),
- EmbeddingBag + stress/yo MLP heads live in ngram_core.onnx,
- _tokenize/_get_positions/exceptions/user-marks/fuse ported 1:1,
- softmax/argmax in numpy (argmax-agreement is the gate; near-ties are
  vanishingly rare and surface as string diffs in verification).
No torch import anywhere in this file.
"""
import json
import os
import re

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RE_COND = r'[^А-Яа-яёЁ]'
VOWELS = 'аоуыэиеяёю'
STRESS = '+'
UNK_ID = 126521


def featurize(words, ngram_dict, unk_id=UNK_ID):
    """words -> (flat_ids int64, offsets int64) exactly like the script
    embedding (modules.py): grams of '<word>' with lengths 1..4 (dict holds
    nothing longer, so longer grams can never hit), UNK fallback, cumulative
    offsets without the trailing total."""
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
    return (np.array(ids, dtype=np.int64),
            np.array(offs[:-1], dtype=np.int64))


def _softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class OnnxNgram:
    def __init__(self, models_dir=None, data_dir=DATA):
        with open(os.path.join(data_dir, "ngrams.json"), encoding="utf-8") as f:
            self.ngram_dict = json.load(f)
        with open(os.path.join(data_dir, "exceptions.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.exceptions = {w: (v[0], v[1]) for w, v in raw.items()}
        self._models_dir = models_dir
        self._sess = None

    def _session(self):
        if self._sess is None:
            from . import env as _env
            from .ort import create_session

            if self._models_dir:
                path = os.path.join(self._models_dir, "ngram_core.onnx")
            else:
                envd = _env.models_dir()
                path = os.path.join(envd, "ngram_core.onnx") if envd else None
            if not path or not os.path.isfile(path):
                raise SystemExit(
                    "silero-onnx: error: missing model file " + str(path) +
                    " (pass models_dir explicitly or set SILERO_ONNX_MODELS)")
            self._sess = create_session(path)
        return self._sess

    # -- rules, ported 1:1 --
    def _tokenize(self, sentence, words_to_ignore=None):
        words_to_ignore = words_to_ignore if words_to_ignore is not None else []
        tokens, model_inputs, prediction_mask = [], [], []
        for word in re.split(r'([\s.,!?;:<>=()/\\]+)', sentence):
            parts = word.split('-')
            if len(parts) == 1:
                cur_tokens = parts
                cur_prediction_mask = [True]
            else:
                cur_tokens = [part + '-' for part in parts[:-1]] + [parts[-1]]
                cur_prediction_mask = [True for _p in parts[:-1]] + [parts[-1] != 'то']
            cur_model_inputs = [re.sub(RE_COND, '', t.lower()) for t in cur_tokens]
            cur_prediction_mask = [((len(x) > 0) and (x not in words_to_ignore)) & m
                                   for x, m in zip(cur_model_inputs, cur_prediction_mask)]
            tokens.extend(cur_tokens)
            model_inputs.extend(cur_model_inputs)
            prediction_mask.extend(cur_prediction_mask)
        return tokens, model_inputs, prediction_mask

    def _get_model_preds(self, words):
        ind, offs = featurize(words, self.ngram_dict)
        s_logits, y_logits = self._session().run(None, {"ind": ind, "offsets": offs})
        s_probs = _softmax(np.asarray(s_logits, dtype=np.float64)).astype(np.float32)
        y_probs = _softmax(np.asarray(y_logits, dtype=np.float64)).astype(np.float32)
        return s_probs, s_probs.argmax(axis=1), y_probs, y_probs.argmax(axis=1)

    def _get_positions(self, word, stressed_vowel_ids, yo_vowel_ids):
        vowel_ids = [i for i, c in enumerate(word) if c in VOWELS]
        ye_ids = [i for i, c in enumerate(word) if c == 'е']
        stress_positions = [vowel_ids[i] for i in stressed_vowel_ids
                            if (i < len(vowel_ids)) and (len(vowel_ids) > 0)]
        yo_positions = [ye_ids[i - 1] for i in yo_vowel_ids
                        if (i > 0) and (i - 1 < len(ye_ids)) and (len(ye_ids) > 0)]
        num_vowels = len(vowel_ids)
        first_vowel_pos = vowel_ids[0] if len(vowel_ids) > 0 else -1
        return stress_positions, yo_positions, num_vowels, first_vowel_pos

    def _accentuate_exception(self, clean_word, raw_word, have_stress):
        exc_stress, exc_yo = self.exceptions[clean_word]
        if have_stress:
            user_pos = [i for i, c in enumerate(raw_word) if c == STRESS]
            out = raw_word.replace(STRESS, '')
            if exc_yo != -1 and (exc_yo + 1 in user_pos):
                out = out[:exc_yo] + ('ё' if out[exc_yo].islower() else 'Ё') + out[(exc_yo + 1):]
            for p in user_pos:
                out = out[:p] + STRESS + out[p:]
            return out
        else:
            if exc_yo != -1:
                raw_word = raw_word[:exc_yo] + ('ё' if raw_word[exc_yo].islower() else 'Ё') + raw_word[(exc_yo + 1):]
            return raw_word[:exc_stress] + STRESS + raw_word[exc_stress:]

    def __call__(self, sentence, put_stress=True, put_yo=True, stress_single_vowel=True,
                 skip_stress_words=None, skip_yo_words=None, words_to_ignore=None):
        skip_stress_words = skip_stress_words if skip_stress_words is not None else []
        skip_yo_words = skip_yo_words if skip_yo_words is not None else []
        if not (put_stress or put_yo):
            return sentence
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence, words_to_ignore)
        s_probs, s_preds, y_probs, y_preds = self._get_model_preds(clean_tokens)
        accented = []
        for wi, (raw_word, clean_word, need) in enumerate(zip(raw_tokens, clean_tokens,
                                                              prediction_mask)):
            raw_lower = raw_word.lower()
            if not need:
                accented.append(raw_word)
                continue
            have_stress = STRESS in raw_lower
            have_yo = 'ё' in raw_lower
            if have_stress and have_yo:
                accented.append(raw_word)
                continue
            if (not have_stress) and have_yo and put_stress:
                if (sum(c in VOWELS for c in raw_lower) == 1 and not stress_single_vowel) \
                        or clean_word.replace('ё', 'е') in skip_stress_words:
                    accented.append(raw_word)
                    continue
                yo_pos = [i for i, x in enumerate(raw_lower) if x == 'ё']
                for i, p in enumerate(yo_pos):
                    raw_word = raw_word[:(p + i)] + STRESS + raw_word[(p + i):]
                accented.append(raw_word)
                continue
            if clean_word in self.exceptions:
                accented.append(self._accentuate_exception(clean_word, raw_word, have_stress))
                continue
            stressed_ids = [int(s_preds[wi])]
            passed_trs = float(s_probs[wi][stressed_ids[0]]) > (0.5 if put_stress else 1)
            set_stress = passed_trs and not have_stress and \
                (clean_word.replace('ё', 'е') not in skip_stress_words)
            yo_ids = [int(y_preds[wi])] if y_preds is not None else [-10]
            passed_yo = y_preds is not None and \
                float(y_probs[wi][yo_ids[0]]) > (0.5 if put_yo else 1)
            set_yo = passed_yo and (clean_word.replace('ё', 'е') not in skip_yo_words)
            if have_stress:
                stressed_ids = [sum(map(part.count, VOWELS))
                                for part in raw_lower.split(STRESS)]
            s_pos, y_pos, n_vow, first_pos = self._get_positions(
                raw_lower, stressed_ids, yo_ids)
            if n_vow == 0:
                accented.append(raw_word)
                continue
            for p in y_pos:
                if p in s_pos and set_yo:
                    if raw_lower[p] == 'е':
                        raw_word = raw_word[:p] + ('ё' if raw_word[p].islower() else 'Ё') + raw_word[(p + 1):]
            if n_vow == 1:
                s_pos = [first_pos]
                set_stress = stress_single_vowel and put_stress
            if not have_stress and set_stress:
                for i, p in enumerate(s_pos):
                    raw_word = raw_word[:(p + i)] + STRESS + raw_word[(p + i):]
            accented.append(raw_word)
        return ''.join(accented).replace('-', '-')
