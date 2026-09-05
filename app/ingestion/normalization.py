"""Нормализация текста перед чанкингом.

Задача — убрать артефакты извлечения (переносы строк внутри абзаца, мягкие
переносы слов, неразрывные пробелы), не разрушив структуру абзацев: границы
абзацев несут смысл для structure-aware чанкинга.
"""

import re
import unicodedata

_SOFT_HYPHEN = "­"
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t   ]+")


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).replace(_SOFT_HYPHEN, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Слово, разорванное переносом строки, склеиваем обратно.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    # Одиночный перенос — это перенос строки внутри абзаца, а не новый абзац.
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
