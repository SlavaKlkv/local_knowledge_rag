"""Датасет для оценки качества retrieval.

Эталон размечается по документам, а не по чанкам: chunk_id пересоздаётся при
каждой переиндексации (другой размер окна, другой парсер), и разметка,
привязанная к чанкам, протухала бы после первой же смены настроек chunking.
Документ же остаётся тем же самым, поэтому такой датасет переживает изменения
пайплайна и позволяет сравнивать конфигурации между собой.

Градация релевантности (grade) нужна nDCG: «документ прямо отвечает на
вопрос» и «документ упоминает тему вскользь» — не одно и то же, и метрика
должна их различать.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import ValidationError


@dataclass(slots=True, frozen=True)
class RelevanceLabel:
    document_id: str
    grade: int = 1


@dataclass(slots=True, frozen=True)
class EvaluationExample:
    """Один размеченный вопрос.

    `has_answer=False` — вопрос, на который в базе знаний ответа нет. Такие
    примеры обязаны быть в датасете: без них метрики поощряют систему,
    которая всегда что-нибудь возвращает.
    """

    id: str
    question: str
    knowledge_base_id: str
    relevant: tuple[RelevanceLabel, ...] = ()
    expected_answer: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def has_answer(self) -> bool:
        return bool(self.relevant)

    @property
    def relevant_ids(self) -> set[str]:
        return {label.document_id for label in self.relevant}

    def grade_of(self, document_id: str) -> int:
        for label in self.relevant:
            if label.document_id == document_id:
                return label.grade
        return 0


@dataclass(slots=True)
class EvaluationDataset:
    examples: list[EvaluationExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def answerable(self) -> list[EvaluationExample]:
        return [example for example in self.examples if example.has_answer]

    @property
    def unanswerable(self) -> list[EvaluationExample]:
        return [example for example in self.examples if not example.has_answer]


def _parse_example(raw: dict, index: int) -> EvaluationExample:
    where = f"пример #{index}"
    for required in ("id", "question", "knowledge_base_id"):
        if not str(raw.get(required, "")).strip():
            raise ValidationError(f"{where}: отсутствует обязательное поле '{required}'")

    labels: list[RelevanceLabel] = []
    for item in raw.get("relevant", []):
        # Короткая форма "doc-id" и полная {"document_id": ..., "grade": ...}:
        # большая часть разметки бинарная, и заставлять писать grade=1 руками
        # значит только провоцировать опечатки.
        if isinstance(item, str):
            labels.append(RelevanceLabel(document_id=item))
            continue
        document_id = str(item.get("document_id", "")).strip()
        if not document_id:
            raise ValidationError(f"{where}: в relevant нет document_id")
        grade = item.get("grade", 1)
        if not isinstance(grade, int) or grade < 1:
            raise ValidationError(
                f"{where}: grade должен быть целым числом >= 1, получено {grade!r}"
            )
        labels.append(RelevanceLabel(document_id=document_id, grade=grade))

    if len({label.document_id for label in labels}) != len(labels):
        raise ValidationError(f"{where}: один и тот же документ размечен дважды")

    return EvaluationExample(
        id=str(raw["id"]),
        question=str(raw["question"]),
        knowledge_base_id=str(raw["knowledge_base_id"]),
        relevant=tuple(labels),
        expected_answer=raw.get("expected_answer"),
        tags=tuple(raw.get("tags", ())),
    )


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Читает датасет из JSON-файла: список примеров или {"examples": [...]}."""
    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"Файл датасета не найден: {file_path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Датасет {file_path} не является корректным JSON: {exc}") from exc

    items = raw.get("examples") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValidationError("Ожидался список примеров или объект с ключом 'examples'")

    examples = [_parse_example(item, index) for index, item in enumerate(items)]
    seen: set[str] = set()
    for example in examples:
        if example.id in seen:
            raise ValidationError(f"Дублирующийся id примера: {example.id}")
        seen.add(example.id)
    return EvaluationDataset(examples=examples)
