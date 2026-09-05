"""Evaluator сравнивает конфигурации retrieval на одном датасете."""

import pytest

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationDataset, EvaluationExample, RelevanceLabel
from app.evaluation.retrieval import RetrievalEvaluator
from app.rag.vector_store import RetrievedChunk


class ScriptedRetriever:
    """Отдаёт заранее заданную выдачу по id вопроса и считает вызовы."""

    def __init__(self, outputs: dict[str, list[str]]) -> None:
        self._outputs = outputs
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query):
        self.calls.append((query.text, query.top_k))
        document_ids = self._outputs.get(query.text, [])
        return [
            RetrievedChunk(
                chunk_id=f"{document_id}-{index}",
                document_id=document_id,
                document_name=f"{document_id}.pdf",
                text="текст",
                score=1.0 - index / 100,
                chunk_index=index,
            )
            for index, document_id in enumerate(document_ids)
        ][: query.top_k]


def dataset(*examples: EvaluationExample) -> EvaluationDataset:
    return EvaluationDataset(examples=list(examples))


def answerable(question: str, *documents: str) -> EvaluationExample:
    return EvaluationExample(
        id=question, question=question, knowledge_base_id="kb-1",
        relevant=tuple(RelevanceLabel(d) for d in documents),
    )


def test_perfect_retrieval_scores_one_everywhere():
    data = dataset(answerable("вопрос-1", "a"), answerable("вопрос-2", "b"))
    retriever = ScriptedRetriever({"вопрос-1": ["a"], "вопрос-2": ["b"]})

    report = RetrievalEvaluator(retriever, k_values=(1,)).evaluate(data)

    assert report.mrr == pytest.approx(1.0)
    assert report.by_k[1].recall == pytest.approx(1.0)
    assert report.by_k[1].ndcg == pytest.approx(1.0)


def test_retrieval_runs_once_per_example_even_for_several_k():
    data = dataset(answerable("вопрос-1", "a"))
    retriever = ScriptedRetriever({"вопрос-1": ["x", "a"]})

    RetrievalEvaluator(retriever, k_values=(1, 3, 10)).evaluate(data)

    # Один вызов с максимальным k; метрики меньших k — срезы того же списка.
    assert retriever.calls == [("вопрос-1", 10)]


def test_smaller_k_is_a_slice_of_the_same_output():
    data = dataset(answerable("вопрос-1", "a"))
    retriever = ScriptedRetriever({"вопрос-1": ["x", "a"]})

    report = RetrievalEvaluator(retriever, k_values=(1, 2)).evaluate(data)

    assert report.by_k[1].recall == 0.0
    assert report.by_k[2].recall == pytest.approx(1.0)


def test_metrics_are_averaged_across_examples():
    data = dataset(answerable("вопрос-1", "a"), answerable("вопрос-2", "b"))
    retriever = ScriptedRetriever({"вопрос-1": ["a"], "вопрос-2": ["мимо"]})

    report = RetrievalEvaluator(retriever, k_values=(1,)).evaluate(data)

    assert report.by_k[1].recall == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)


def test_unanswerable_examples_do_not_affect_recall():
    """Иначе вопрос без ответа обнулял бы Recall и метрика стала бы нечитаемой."""
    data = dataset(
        answerable("вопрос-1", "a"),
        EvaluationExample("q-no", "чего нет в базе", "kb-1"),
    )
    retriever = ScriptedRetriever({"вопрос-1": ["a"], "чего нет в базе": ["шум"]})

    report = RetrievalEvaluator(retriever, k_values=(1,)).evaluate(data)

    assert report.answerable_total == 1
    assert report.unanswerable_total == 1
    assert report.by_k[1].recall == pytest.approx(1.0)


def test_false_positive_rate_counts_output_for_unanswerable_questions():
    data = dataset(
        EvaluationExample("q-no-1", "нет-1", "kb-1"),
        EvaluationExample("q-no-2", "нет-2", "kb-1"),
    )
    retriever = ScriptedRetriever({"нет-1": ["шум"], "нет-2": []})

    report = RetrievalEvaluator(retriever, k_values=(3,)).evaluate(data)

    assert report.false_positive_rate == pytest.approx(0.5)


def test_false_positive_rate_is_absent_without_unanswerable_examples():
    data = dataset(answerable("вопрос-1", "a"))

    report = RetrievalEvaluator(ScriptedRetriever({"вопрос-1": ["a"]})).evaluate(data)

    assert report.false_positive_rate is None
    assert "false_positive_rate" not in report.as_dict()


def test_report_serializes_every_requested_k():
    data = dataset(answerable("вопрос-1", "a"))
    evaluator = RetrievalEvaluator(ScriptedRetriever({"вопрос-1": ["a"]}), k_values=(3, 1))

    report = evaluator.evaluate(data, name="hybrid").as_dict()

    assert report["retriever"] == "hybrid"
    # k упорядочены по возрастанию независимо от порядка в конфигурации.
    assert [entry["k"] for entry in report["by_k"]] == [1, 3]


def test_empty_dataset_is_rejected():
    with pytest.raises(ValidationError, match="пуст"):
        RetrievalEvaluator(ScriptedRetriever({})).evaluate(dataset())


@pytest.mark.parametrize("k_values", [(), (0,), (3, -1)])
def test_invalid_k_values_are_rejected(k_values):
    with pytest.raises(ValidationError):
        RetrievalEvaluator(ScriptedRetriever({}), k_values=k_values)
