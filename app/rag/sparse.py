"""Sparse-представление текста для лексического поиска.

Dense-эмбеддинги хорошо ловят смысловую близость, но плохо — точные
совпадения (номера статей, коды, редкие термины). Sparse-вектор ловит
именно это и вместе с dense даёт hybrid retrieval.

Полноценный BM25 требует статистики по всей коллекции (IDF), которую нельзя
посчитать на одном документе в изоляции. Вместо этого используется hashing
trick: термин хешируется в индекс фиксированного словаря, вес — сублинейная
частота (1 + log(tf)), что подавляет влияние часто повторяющихся слов
внутри одного чанка, сохраняя точные лексические совпадения при поиске.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_DEFAULT_VOCAB_SIZE = 2**18


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


# Полноценная лемматизация тянет за собой словарь и вторую модель в образ.
# Для сопоставления слов запроса со словами текста хватает усечения самых
# частотных русских окончаний: «командировка», «командировки» и
# «командировок» сходятся к одной основе, а слова разных корней — нет.
_ENDINGS = (
    "ами", "ями", "ого", "ему", "ому", "ыми", "ими", "его", "ей", "ой", "ый",
    "ий", "ая", "яя", "ое", "ее", "ам", "ям", "ах", "ях", "ов", "ев", "ом",
    "ем", "ую", "юю", "ые", "ие", "ых", "их", "ии", "ок", "ек", "ы", "и", "а", "я", "о",
    "е", "у", "ю", "й", "ь",
)
_MIN_STEM = 4


def stem(token: str) -> str:
    """Грубая основа слова: отсечение падежного окончания."""
    for ending in _ENDINGS:
        if len(token) - len(ending) >= _MIN_STEM and token.endswith(ending):
            return token[: -len(ending)]
    return token


def stems(text: str) -> list[str]:
    return [stem(token) for token in tokenize(text)]


class HashedSparseVectorizer:
    """Векторизатор без внешнего словаря: индекс термина = хеш от токена
    по модулю фиксированного размера пространства."""

    def __init__(self, vocab_size: int = _DEFAULT_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.vocab_size

    def vectorize(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        counts = Counter(tokens)
        weighted: dict[int, float] = {}
        for token, count in counts.items():
            index = self._index(token)
            weight = 1.0 + math.log(count)
            # Разные токены могут схлопнуться в один индекс (коллизия
            # хеша) — суммируем вес, а не перезаписываем.
            weighted[index] = weighted.get(index, 0.0) + weight

        indices = sorted(weighted)
        values = [weighted[i] for i in indices]
        return SparseVector(indices=indices, values=values)


def stem_matches(left: str, right: str) -> bool:
    """Совпадают ли две основы с точностью до беглой гласной.

    Усечение окончания не выравнивает пары вроде «командировка» →
    «командировк» и «командировок» → «командиров»: в родительном падеже
    множественного числа появляется беглая «о». Одна основа при этом
    остаётся началом другой, поэтому основы, различающиеся хвостом в один-два
    символа, считаются одним словом.

    Послабление намеренно узкое. Без ограничения на длину хвоста началом
    длинной основы оказывается любая короткая: «командировк» начинается с
    «команд», и запрос про командировки находил бы «дни командных встреч».
    """
    if left == right:
        return True
    if min(len(left), len(right)) < _MIN_STEM:
        return False
    if abs(len(left) - len(right)) > 2:
        return False
    return left.startswith(right) or right.startswith(left)


# Служебные слова несут не смысл запроса, а его грамматику: они есть почти
# в каждом тексте, и по ним «Как оформить ипотеку?» совпало бы с любым
# фрагментом базы через «как». В подсчёте лексического совпадения они не
# участвуют.
_STOP_WORDS = frozenset(
    """
    и а но или да же ли бы не ни в во на за под над из от до по с со к ку у о
    об обо при про для без через между перед после это этот эта эти тот та то
    те так там тут здесь как что чей чем чём кто где когда куда откуда зачем
    почему сколько какой какая какое какие каких каком нужно надо может можно
    есть быть был была было были будет будут я ты он она оно мы вы они мне
    тебе ему ей нам вам им его её их себя свой своя своё свои весь вся всё все
    еще ещё уже
    """.split()  # noqa: SIM905 — читаемый список слов важнее литерала
)


def content_stems(text: str) -> list[str]:
    """Основы значимых слов: без служебных.

    Если значимых слов в запросе не осталось, возвращаются все — запрос
    вроде «что где когда» лучше отработать буквально, чем как пустой.
    """
    tokens = tokenize(text)
    content = [token for token in tokens if token not in _STOP_WORDS]
    return [stem(token) for token in (content or tokens)]
