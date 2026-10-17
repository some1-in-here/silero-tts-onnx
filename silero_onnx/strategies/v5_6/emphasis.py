"""v5_6 emphasis/focus/type front-end (torch-free pure port).

Mirrors snakers4/silero-models ru/v5_6_ru package
(multi_acc_v3_package.py) text algorithms EXACTLY:
  TYPE2ID / classify_sentence / classify_text (question taxonomy),
  **-emphasis validation + ** -> * conversion, *-focus stripping,
  <intensity=N> extraction, focus-word re-marking on accented text,
  parse_focus_from_text, char_sent_ids / token_sent_spans,
  emph_spans_from_focus, build_type_ids_tokens, place_tokens.

Nothing here imports torch (verified like the rest of `silero_onnx`);
tensors are numpy. Deviations from upstream (documented, all loud):
- intensity is validated to 0..3 (focus_embedding has 4 rows; upstream
  would OOB inside the embedding lookup with a cryptic error).
- several <intensity=N> tags: the first wins for the whole utterance
  (upstream keeps intensities[0] too), but differing values warn
  instead of upstream's silent drop.
- bad ** markup raises ValueError with upstream's own message.
- SSML focus path (restore_focus_marks asserts, windowed refinement)
  is NOT ported: the engine uses full-sequence predictor gt (the
  documented operating point), so only plain-text markers are handled.
"""
import re
import warnings

import numpy as np

assert "torch" not in __import__("sys").modules, \
    "torch must never be imported here"

TYPE2ID = {"st": 0, "wh_q": 1, "general_q": 2, "alternative_q": 3,
           "tag_q": 4, "exclam": 5}

WH_FORMS = {
    "кто", "кого", "кому", "кем", "ком",
    "что", "чего", "чему", "чем", "чём", "чё", "чо",
    "где", "куда", "откуда", "когда",
    "почему", "зачем", "как", "почём", "отчего", "насколько",
    "сколько", "скольких", "скольким", "сколькими",
    "какой", "какая", "какое", "какие",
    "какого", "каких", "какому", "каким",
    "какими", "каком", "какую",
    "чей", "чья", "чьё", "чье", "чьи",
    "чьего", "чьей", "чьих",
    "чьему", "чьим", "чьими",
    "чьём", "чьем", "чью",
    "который", "которая", "которое", "которые",
    "которого", "которой", "которых",
    "которому", "которым", "которыми",
    "котором", "которую",
    "каков", "какова", "каково", "каковы",
}

LEADING_FILLERS = {
    "а", "ну", "и", "так", "вот", "слышь", "слушай",
    "скажите", "скажи", "пожалуйста",
    "вобще", "вообще", "типа", "короче",
}

TAG_PATTERNS = [
    r"[,\s]+(правда|верно|да)\s*\?$",
    r"[,\s]+(не\s+так\s+ли|не\s+правда\s+ли|разве\s+не\s+так|ведь\s+так)\s*\?$",
    r"[,\s]+ведь\s*\?$",
    r"[,\s]+а\s*\?$",
]
TAG_RE = re.compile("|".join(f"(?:{p})" for p in TAG_PATTERNS),
                    re.IGNORECASE)

QUOTE_OPEN = {'"', '«', chr(8220), chr(8222)}
QUOTE_CLOSE = {'"', '»', chr(8221), chr(8217)}
WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
STAR_RE = re.compile(r'(\*{1,2})([^*\s]+)\1')
CORE_RE = re.compile(r'[^\W\d_]')
FOCUS_SKIP_CHARS = set('.,!?;:…–')

DEFAULT_INTENSITY = 3


def _strip_outer_quotes(s):
    s = s.strip()
    while s and s[0] in QUOTE_OPEN:
        s = s[1:].strip()
    while s and s[-1] in QUOTE_CLOSE:
        s = s[:-1].strip()
    return s


def _normalize_for_clf(s):
    s = _strip_outer_quotes(s)
    s = s.replace("+", "")
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_wh_prefix(text):
    words = WORD_RE.findall(text.lower())[:8]
    content = []
    for w in words:
        if w in LEADING_FILLERS:
            continue
        content.append(w)
        if len(content) >= 4:
            break
    return any(w in WH_FORMS for w in content)


def classify_sentence(s):
    s = s.strip()
    if not s:
        return "st"
    s_clean = _strip_outer_quotes(s)
    if not s_clean:
        return "st"
    tail = s_clean
    if tail.endswith("?!") or tail.endswith("?..") or tail.endswith("?..."):
        tail = tail.rstrip(".!")
    if tail.endswith("?"):
        q = _normalize_for_clf(tail)
        if TAG_RE.search(q):
            return "tag_q"
        if _has_wh_prefix(q):
            return "wh_q"
        if re.search(r"\bили\b", q, re.IGNORECASE):
            return "alternative_q"
        return "general_q"
    if s_clean.endswith("!"):
        return "exclam"
    return "st"


def classify_text(text):
    if not text or not text.strip():
        return "st"
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text.strip())
                 if s.strip()]
    types = [classify_sentence(s) for s in sentences]
    return "|".join(types)


def extract_intensity(text):
    """<intensity=N>...</intensity> -> (cleaned, level); default 3.

    Upstream's only channel is ssml_text, where the tag is stripped
    before XML parsing (extract_intensity_from_ssml); SSML mode here
    does the same, and plain text accepts the tags too (superset). The
    level scales *focus* marks only (see parse_focus_from_text): with
    no * / ** marks it has no effect — upstream audio is identical.
    Several tags: the first one wins for the whole utterance (upstream
    keeps intensities[0] too), but differing values warn — upstream
    drops the rest silently."""
    intensities = re.findall(r'<intensity=(\d+)>', text)
    cleaned = re.sub(r'<intensity=(\d+)>', '', text)
    cleaned = cleaned.replace('</intensity>', '')
    level = int(intensities[0]) if intensities else DEFAULT_INTENSITY
    if not 0 <= level <= 3:
        raise ValueError(
            f"silero-onnx: error: intensity must be 0..3 (focus embedding "
            f"has 4 rows), got {level}")
    if len(set(intensities)) > 1:
        warnings.warn(
            "silero-onnx: multiple <intensity=N> values "
            f"({', '.join(intensities)}): upstream applies the first one "
            f"({level}) to the whole utterance and drops the rest")
    return cleaned, level


def strip_markup_tags(text):
    """Remaining <...> -> single space (upstream strip_ssml_tags)."""
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text)).strip()


def split_emphasis(text):
    """**-emphasis validation + conversion (upstream apply_tts :661-674).

    Returns (text_with_single_stars, emph_idx, has_emphasis); raises
    ValueError with upstream's message on malformed markup."""
    has_emphasis = False
    emph_idx = set()
    if '**' in text:
        matches = re.findall(r'\*\*([^*\s]+)\*\*', text)
        if text.count('**') != 2 * len(matches):
            raise ValueError("Некорректная разметка эмфазы. "
                             "Ожидается формат **слово**.")
        if matches:
            has_emphasis = True
            emph_idx = {i for i, m in enumerate(STAR_RE.finditer(text))
                        if len(m.group(1)) == 2}
            text = re.sub(r'\*\*([^*\s]+)\*\*', r'*\1*', text)
    return text, emph_idx, has_emphasis


def clean_focus_word(word):
    return re.sub(r'[\W\d_]+', '', word.lower().replace('+', ''))


def mark_word(word):
    """Stars wrap the word core; punctuation stays outside (upstream)."""
    pos = [i for i, ch in enumerate(word) if ch == '+' or CORE_RE.match(ch)]
    a, b = pos[0], pos[-1] + 1
    return f'{word[:a]}*{word[a:b]}*{word[b:]}'


def apply_focus_marks(accented_text, original_marked_text):
    """Re-mark accented text by word matching (upstream :743-755).

    focus_words come from the *-marked raw text (stars/plus stripped,
    lowered); each is located once, in order, in the accented words and
    re-wrapped with mark_word. Returns the re-marked accented string."""
    focus_words = []
    for word in original_marked_text.split():
        if '*' in word:
            focus_words.append(word.replace('*', '').replace('+', '')
                               .lower())
    out = []
    for word in accented_text.split():
        clean_word = word.replace('+', '').lower()
        if clean_word in focus_words:
            out.append('*' + word + '*')
            focus_words.remove(clean_word)
        else:
            out.append(word)
    return ' '.join(out)


def parse_focus_from_text(text, seq_len, symbol_to_id, intensity=3):
    """*-marked accented text -> focus mask (1, seq_len) int64.

    Upstream PartTTSModelMultiAcc_v3.parse_focus_from_text verbatim,
    torch replaced with numpy: per-char focus flags (toggled by *),
    focused non-punctuation symbols take `intensity`, rest 0, laid out
    with place_tokens (sos at [0,0], tail at [0,-1])."""
    level = int(intensity)
    focus_flags_raw = []
    in_focus = False
    for ch in text:
        if ch == '*':
            in_focus = not in_focus
            continue
        focus_flags_raw.append(in_focus)
    clean_text = text.replace('*', '')
    processed = clean_text.lower().replace('-', ' ')
    values = []
    for i, ch in enumerate(processed):
        if ch in symbol_to_id:
            flag = focus_flags_raw[i] if i < len(focus_flags_raw) else False
            values.append(level if flag and ch not in FOCUS_SKIP_CHARS
                          else 0)
    return clean_text, place_tokens(values, seq_len, symbol_to_id=symbol_to_id)


def char_sent_ids(text):
    """Separator whitespace belongs to the preceding sentence (upstream)."""
    ids = []
    pos, sent_idx = 0, 0
    for m in SENT_SPLIT_RE.finditer(text):
        ids.extend([sent_idx] * (m.end() - pos))
        pos = m.end()
        sent_idx += 1
    ids.extend([sent_idx] * (len(text) - pos))
    return ids


def token_sent_spans(sequence, id_to_symbol):
    """Sentence spans over the token sequence (sos/eos/edge spaces cut).

    Returns (spans, sent_texts); mirrors upstream token_sent_spans."""
    toks = [id_to_symbol[i] for i in sequence]
    sent_ids = char_sent_ids(''.join(toks[1:-1]))
    spans = []
    for k in range(sent_ids[-1] + 1 if sent_ids else 0):
        idx = [i + 1 for i, s in enumerate(sent_ids) if s == k]
        a, b = idx[0], idx[-1] + 1
        while b > a and toks[b - 1] == ' ':
            b -= 1
        spans.append((a, b))
    return spans, [''.join(toks[a:b]) for a, b in spans]


def emph_spans_from_focus(spans, fm_list, emph_idx):
    """Focus-word order -> sentence set holding ** groups (upstream)."""
    span_of = {p: k for k, (a, b) in enumerate(spans) for p in range(a, b)}
    out, g = set(), -1
    for p in range(1, len(fm_list) - 1):
        if fm_list[p] > 0 and (p == 1 or fm_list[p - 1] == 0):
            g += 1
            if g in emph_idx and p in span_of:
                out.add(span_of[p])
    return out


def place_tokens(values, seq_len, symbol_to_id=None, sos_token='|', tail=0):
    """Token-layout writer (upstream place_tokens, numpy).

    values[i] -> position i+1 (sos slot); [0,0] takes values[0],
    [0,-1] takes tail. Empty values -> all zeros."""
    out = np.zeros((1, seq_len), dtype=np.int64)
    if not values:
        return out
    offset = 1 if sos_token != '' else 0
    for i, v in enumerate(values):
        j = i + offset
        if j < seq_len - offset:
            out[0, j] = v
    if offset:
        out[0, 0] = values[0]
        out[0, -1] = tail
    return out


def build_type_ids_tokens(sequence, types_list, emph_spans, id_to_symbol,
                           sos_token='|'):
    """Per-token sentence types (+6 on emphasis spans), sos/eos aware."""
    tids = [TYPE2ID.get(t, 0) + (len(TYPE2ID) if k in emph_spans else 0)
            for k, t in enumerate(types_list)]
    toks = [id_to_symbol[i] for i in sequence]
    sent_ids = char_sent_ids(''.join(toks[1:-1]))
    values = [tids[min(s, len(tids) - 1)] for s in sent_ids]
    return place_tokens(values, len(toks), sos_token=sos_token,
                        tail=values[-1] if values else 0)


def locate_emph_sents(marked_text, emph_idx):
    """Sentences holding ** star groups, markup order (upstream)."""
    out, k = set(), 0
    for si, s in enumerate(SENT_SPLIT_RE.split(marked_text.strip())):
        for _ in STAR_RE.finditer(s):
            if k in emph_idx:
                out.add(si)
            k += 1
    return out
