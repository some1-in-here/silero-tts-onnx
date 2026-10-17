"""SSML frontend (pure Python, stdlib only): <speak> XML to segments.

Faithful port of the upstream v5 package parsing (process_ssml*):
<speak> root is mandatory; <break time|strength>; <prosody rate|pitch>;
<p>/<s> imply strong/x-strong breaks. Output is a list of segments:
{"text", "rate", "pitch", "break_ms" or None}.

Downstream split (by design, mirrors upstream dataflow):
- <prosody rate> becomes a per-token durs_rate array (the acoustic graph
  already takes it; no graph changes); <prosody pitch> needs a
  pitch_coefs input that only the v5_ru acoustic graph has — v5_6_ru
  warns and ignores it (its pitch comes from the predictor instead;
  never a silent drop);
- <break> becomes a (token_pos, frames) plan applied by the shared
  two-pass path in strategies/pauses.py (the graphs have no symb_durs
  input).

Deliberate deviations from upstream (all documented, none silent):
- per-segment text goes through OUR normalize() (abbr/numbers/case),
  not upstream prepare_text_input (which drops digits instead of
  expanding them) — ours is strictly more capable;
- has_text = segment holds non-space chars (upstream filters harder);
- errors are ValueError (upstream mixes AssertionError/ValueError);
- fractional seconds ("1.5s") fail here exactly as upstream
  (int() parsing), pinned by test.
"""
import re
import warnings
import xml.etree.ElementTree as ET

STRENGTH2TIME = {'x-weak': 25, 'weak': 75, 'medium': 150,
                 'strong': 300, 'x-strong': 1000}
RATE2VALUE = {'x-slow': 0.5, 'slow': 0.8, 'medium': 1.,
              'fast': 1.2, 'x-fast': 1.5}
PITCH2VALUE = {'x-low': 0.75, 'low': 0.85, 'medium': 1., 'high': 1.15,
               'x-high': 1.25, 'robot': 0.}
MAX_BREAK_MS = 5000
VALID_BREAK = list(STRENGTH2TIME)
VALID_RATE = list(RATE2VALUE)
VALID_PITCH = list(PITCH2VALUE)


def parse_break_attrib(attrib):
    """{'strength'|'time'} -> (strength, ms). Mirrors process_break_attrib."""
    for k in attrib.keys():
        if k not in ('strength', 'time'):
            warnings.warn(f"silero-onnx: unsupported SSML <break> attrib: {k}")
    strength = attrib.get('strength', 'medium')
    break_time = attrib.get('time', None)
    if break_time is not None:
        if break_time.endswith('ms'):
            break_ts = int(break_time[:-2])
        elif break_time.endswith('s'):
            break_ts = int(break_time[:-1]) * 1000
        else:
            raise ValueError("Invalid <break> tag, time should end with "
                             "'ms' or 's'")
        if break_ts >= STRENGTH2TIME['x-strong']:
            strength = 'x-strong'
        elif break_ts >= STRENGTH2TIME['strong']:
            strength = 'strong'
    else:
        if strength in STRENGTH2TIME:
            break_ts = STRENGTH2TIME[strength]
        else:
            raise ValueError(f"Invalid <break> tag, strength should be in "
                             f"{', '.join(VALID_BREAK)}")
    if break_ts > MAX_BREAK_MS:
        warnings.warn('silero-onnx: pauses over 5 sec are clamped to 5 sec')
        break_ts = MAX_BREAK_MS
    return strength, break_ts


def parse_prosody(attrib):
    """{'rate'|'pitch'} -> (rate, pitch, change_rate, change_pitch)."""
    for k in attrib.keys():
        if k not in ('rate', 'pitch'):
            warnings.warn(f"silero-onnx: unsupported SSML <prosody> attrib: {k}")
    rate = attrib.get('rate', None)
    pitch = attrib.get('pitch', None)
    if rate is None and pitch is None:
        raise ValueError("Empty <prosody> tag")
    if rate is not None:
        change_rate = True
        if rate.endswith('%'):
            rate_val = int(rate.replace('%', '')) / 100
        else:
            rate_val = RATE2VALUE.get(rate, None)
            if rate_val is None:
                raise ValueError(f"Invalid <prosody> tag, rate should be in "
                                 f"{', '.join(VALID_RATE)}")
    else:
        change_rate = False
        rate_val = 1.
    if pitch is not None:
        change_pitch = True
        if pitch.endswith('%'):
            pitch_val = int(pitch.replace('%', '')[1:]) / 100
            if pitch[0] == '+':
                pitch_val = 1. + pitch_val
            else:
                pitch_val = 1. - pitch_val
        else:
            pitch_val = PITCH2VALUE.get(pitch, None)
            if pitch_val is None:
                raise ValueError(f"Invalid <prosody> tag, pitch should be in "
                                 f"{', '.join(VALID_PITCH)}")
    else:
        change_pitch = False
        pitch_val = 1.
    return rate_val, pitch_val, change_rate, change_pitch


def _head_tail(text, rate, pitch):
    if text is None:
        return []
    proc = re.sub(r'\s+', ' ', text.replace('\n', '')).strip()
    return [{'text': proc,
             'break': {'strength': None, 'time': None},
             'prosody': {'rate': rate, 'pitch': pitch}}]


def _element(el, strength, rate, pitch):
    parsed, last_tag = [], None
    parsed.extend(_head_tail(el.text, rate, pitch))
    for child in el:
        if child.tag == 'break':
            b_strength, b_ts = parse_break_attrib(child.attrib)
            if len(parsed) == 0:
                parsed.append({'text': '.',
                               'break': {'strength': None, 'time': None},
                               'prosody': {'rate': rate, 'pitch': pitch}})
            parsed[-1]['break'] = {'strength': b_strength, 'time': b_ts}
        elif child.tag == 'prosody':
            p_rate, p_pitch, ch_rate, ch_pitch = parse_prosody(child.attrib)
            child_parsed = _element(
                child, strength,
                p_rate if ch_rate else rate,
                p_pitch if ch_pitch else pitch)
            parsed.extend(child_parsed)
        elif child.tag in ('p', 's'):
            b_strength = 'strong' if child.tag == 's' else 'x-strong'
            child_parsed = _element(child, b_strength, rate, pitch)
            if len(parsed) > 0 and (parsed[-1]['text'] or last_tag is not None
                                    or last_tag != child.tag):
                if parsed[-1]['break']['strength'] is None:
                    parsed[-1]['break'] = {'strength': b_strength,
                                           'time': STRENGTH2TIME[b_strength]}
                else:
                    last_time = parsed[-1]['break']['time']
                    parsed[-1]['break'] = {
                        'strength': b_strength,
                        'time': max(last_time, STRENGTH2TIME[b_strength])}
            if len(child_parsed) > 0:
                if child_parsed[-1]['break']['strength'] is None:
                    child_parsed[-1]['break'] = {
                        'strength': b_strength,
                        'time': STRENGTH2TIME[b_strength]}
                else:
                    last_time = child_parsed[-1]['break']['time']
                    child_parsed[-1]['break'] = {
                        'strength': b_strength,
                        'time': max(last_time, STRENGTH2TIME[b_strength])}
            parsed.extend(child_parsed)
        else:
            warnings.warn(f"silero-onnx: unsupported SSML tag: {child.tag} "
                          f"(its text is dropped, tail kept)")
        last_tag = child.tag
        if child.tail:
            tail_text = child.tail
            if tail_text and tail_text[0] in '.,!?…–;:' and len(parsed) > 0:
                parsed[-1]['text'] = parsed[-1]['text'].strip() + tail_text[0]
                if len(tail_text) > 1:
                    tail_text = tail_text[1:]
            parsed.extend(_head_tail(tail_text, rate, pitch))
    return parsed


def _check_break(text, tbreak):
    if len(text) == 0 or (tbreak['strength'] is not None
                          and text[-1] not in '!,-.:;?–…'):
        text = text + '.'
    return text


def parse_ssml(ssml_text):
    """<speak> XML -> [{'text', 'rate', 'pitch', 'break_ms'|None}]."""
    ssml_text = re.sub(r'\s+', ' ', ssml_text).strip().replace('\n ', '\n')
    try:
        root = ET.fromstring(ssml_text)
    except Exception as e:
        detail = str(e) or "malformed XML"
        hint = ""
        close = ssml_text.rfind("</speak>")
        if close != -1 and ssml_text[close + len("</speak>"):].strip():
            hint = (" Text after </speak> is not allowed — wrap everything "
                    "in a single <speak>...</speak> block.")
        elif "<speak" not in ssml_text:
            hint = (" SSML must be a single <speak>...</speak> block "
                    "(the <speak> root is mandatory).")
        raise ValueError(f"Invalid SSML ({detail}).{hint}")
    if root.tag != 'speak':
        raise ValueError("Invalid SSML format: <speak> tag is essential")
    try:
        items = _element(root, 'strong', 1., 1.)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse SSML: {e}")
    segments = []
    for i, item in enumerate(items):
        tbreak, tprosody = item['break'], item['prosody']
        text = item['text']
        has_text = bool(re.sub(r'\s+', '', text))
        break_ms = tbreak['time']
        if has_text or i == 0:
            text = _check_break(text, tbreak)
            segments.append({'text': text,
                             'rate': tprosody['rate'],
                             'pitch': tprosody['pitch'],
                             'break_ms': break_ms})
        elif tbreak['strength'] is not None and len(segments) > 0:
            prev = segments[-1]
            prev['text'] = _check_break(prev['text'], tbreak)
            if prev['break_ms'] is None:
                prev['break_ms'] = break_ms
            else:
                prev['break_ms'] = max(break_ms, prev['break_ms'])
    return segments


def ms_to_frames(ms):
    """Break milliseconds -> duration frames (80 Hz frame clock)."""
    return int(ms / 12.5)
