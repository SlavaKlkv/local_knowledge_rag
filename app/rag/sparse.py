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
