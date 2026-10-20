"""Russian abbreviation expansion (moved from frontend.py unchanged)."""
import re

ABBR = {
    # street addresses
    "ул.": "улица", "пр.": "проспект", "пер.": "переулок", "пл.": "площадь",
    "наб.": "набережная", "бул.": "бульвар", "ш.": "шоссе", "туп.": "тупик",
    "д.": "дом", "корп.": "корпус", "кв.": "квартира",
    "оф.": "офис", "эт.": "этаж", "под.": "подъезд",
    # geography / admin
    "с.": "село", "дер.": "деревня", "пос.": "посёлок",
    "р.": "река", "оз.": "озеро", "обл.": "область", "р-н": "район",
    "ст.": "станция",
    # misc frequent
    "т.д.": "так далее", "т.е.": "то есть", "т.к.": "так как",
    "т.н.": "так называемый", "др.": "другой", "им.": "имени",
    "гг.": "годы", "г.г.": "годы", "в.": "век", "вв.": "века",
    "рис.": "рисунок", "табл.": "таблица",
    "руб.": "рублей", "коп.": "копеек", "шт.": "штук",
    "кг.": "килограмм", "гр.": "грамм", "мл.": "миллилитров",
    "л.": "литров", "км.": "километров",
    "мм.": "миллиметров",
    "ч.": "часов", "мин.": "минут", "сек.": "секунд",
}
# Ambiguous ones, resolved by local context (documented heuristic):
#   "стр. 5" (digit follows) -> страница, else строение
#   "г. Москва" (capitalized word follows) -> город, else год
#   "м. Тверская" (known station context is undecidable locally) -> метро
#   only before capitalized words, else метров
#   "с." kept as село (page sense "с." is rare in speech input).
_AMBIG = (
    (re.compile(r"\bстр\.(?=\s*\d)", flags=re.IGNORECASE), "страница"),
    (re.compile(r"\bстр\.", flags=re.IGNORECASE), "строение"),
    (re.compile(r"\bг\.(?=\s*[А-ЯЁ])"), "город"),
    (re.compile(r"\bг\.", flags=re.IGNORECASE), "год"),
    (re.compile(r"\bм\.(?=\s*[А-ЯЁ])"), "метро"),
    (re.compile(r"\bм\.", flags=re.IGNORECASE), "метров"),
    # "См." (смотри) vs "5 см" (сантиметров): digits before -> unit
    (re.compile(r"(?<=\d\s)см\.", flags=re.IGNORECASE), "сантиметров"),
    (re.compile(r"\bсм\.", flags=re.IGNORECASE), "смотри"),
)
_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ABBR), key=len,
                                                  reverse=True)) + r")",
                      flags=re.IGNORECASE)


def expand_abbreviations(text: str) -> str:
    """Replace known Russian abbreviations (case-insensitive, dot-aware).

    Ambiguous forms (стр./г./м.) are resolved by the following token
    (documented heuristic, see _AMBIG); case of the first letter kept.
    """
    def keep_case(m, val):
        return val[0].upper() + val[1:] if m.group(0)[:1].isupper() else val
    for rx, val in _AMBIG:
        text = rx.sub(lambda m: keep_case(m, val), text)

    def rep(m):
        key = m.group(0).lower()
        val = ABBR.get(key, m.group(0))
        return val[0].upper() + val[1:] if m.group(0)[:1].isupper() else val
    return _ABBR_RE.sub(rep, text)
