import json

import pytest

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationExample, RelevanceLabel, load_dataset


def write(tmp_path, payload) -> str:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def example(**overrides) -> dict:
    base = {
        "id": "q1",
        "question": "Сколько дней отпуска?",
        "knowledge_base_id": "kb-1",
        "relevant": ["doc-1"],
    }
    base.update(overrides)
    return base


def test_short_and_full_relevance_forms_are_equivalent(tmp_path):
    dataset = load_dataset(
        write(
            tmp_path,
            [
                example(id="q1", relevant=["doc-1"]),
                example(id="q2", relevant=[{"document_id": "doc-1", "grade": 1}]),
            ],
        )
    )

    assert dataset.examples[0].relevant == dataset.examples[1].relevant


def test_grades_are_preserved(tmp_path):
    dataset = load_dataset(
        write(
            tmp_path,
            [example(relevant=[{"document_id": "doc-1", "grade": 3}, "doc-2"])],
        )
    )

    loaded = dataset.examples[0]
    assert loaded.grade_of("doc-1") == 3
    assert loaded.grade_of("doc-2") == 1
    assert loaded.grade_of("doc-неизвестный") == 0


def test_dataset_accepts_object_with_examples_key(tmp_path):
    dataset = load_dataset(write(tmp_path, {"examples": [example()]}))

    assert len(dataset) == 1


def test_example_without_relevant_documents_is_unanswerable(tmp_path):
    dataset = load_dataset(
        write(tmp_path, [example(id="q1"), example(id="q2", relevant=[])])
    )

    assert [e.id for e in dataset.answerable] == ["q1"]
    assert [e.id for e in dataset.unanswerable] == ["q2"]


@pytest.mark.parametrize("field", ["id", "question", "knowledge_base_id"])
def test_missing_required_field_is_rejected(tmp_path, field):
    payload = example()
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        load_dataset(write(tmp_path, [payload]))


def test_duplicate_example_ids_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="Дублирующийся id"):
        load_dataset(write(tmp_path, [example(), example()]))


def test_duplicate_document_in_one_example_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="дважды"):
        load_dataset(write(tmp_path, [example(relevant=["doc-1", "doc-1"])]))


@pytest.mark.parametrize("grade", [0, -1, "высокая", 1.5])
def test_invalid_grade_is_rejected(tmp_path, grade):
    payload = example(relevant=[{"document_id": "doc-1", "grade": grade}])

    with pytest.raises(ValidationError, match="grade"):
        load_dataset(write(tmp_path, [payload]))


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ValidationError, match="не найден"):
        load_dataset(tmp_path / "нет-такого.json")


def test_broken_json_is_reported_clearly(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text("{не json", encoding="utf-8")

    with pytest.raises(ValidationError, match="JSON"):
        load_dataset(path)


def test_has_answer_follows_from_relevance():
    assert EvaluationExample("q", "?", "kb", (RelevanceLabel("d"),)).has_answer is True
    assert EvaluationExample("q", "?", "kb").has_answer is False
