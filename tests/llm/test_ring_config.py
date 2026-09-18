"""Хранилище пользовательского состава кольца."""

import json

import pytest

from app.core.errors import ValidationError
from app.hardware.profiles import HardwareProfile, get_profile_definition
from app.llm.ring_config import RingConfigStore, is_embedding_model


def _store(tmp_path) -> RingConfigStore:
    return RingConfigStore(tmp_path / "ring.json")


def _profile_ring():
    return get_profile_definition(HardwareProfile.LIGHT).ring


def test_untouched_store_falls_back_to_the_profile_ring(tmp_path):
    resolved = _store(tmp_path).resolve(_profile_ring())

    assert [e.model for e in resolved] == [e.model for e in _profile_ring()]


def test_saved_order_is_the_traversal_order(tmp_path):
    store = _store(tmp_path)
    store.save(["llama3.1:8b", "qwen3:4b"])

    assert [e.model for e in store.resolve(_profile_ring())] == [
        "llama3.1:8b",
        "qwen3:4b",
    ]


def test_model_from_another_profile_keeps_its_catalog_data(tmp_path):
    store = _store(tmp_path)
    store.save(["qwen3:32b"])

    (entry,) = store.resolve(_profile_ring())
    assert entry.family == "qwen"
    assert entry.download_size_gb > 0


def test_model_outside_the_catalog_gets_family_from_its_name(tmp_path):
    store = _store(tmp_path)
    store.save(["mistral:7b"])

    (entry,) = store.resolve(_profile_ring())
    assert entry.family == "mistral"
    assert entry.download_size_gb == 0


def test_empty_composition_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        _store(tmp_path).save(["  "])


def test_duplicate_model_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        _store(tmp_path).save(["qwen3:4b", "qwen3:4b"])


def test_clear_returns_the_ring_to_the_profile(tmp_path):
    store = _store(tmp_path)
    store.save(["llama3.1:8b"])
    store.clear()

    assert store.load() is None


def test_broken_file_degrades_to_the_profile_ring(tmp_path):
    path = tmp_path / "ring.json"
    path.write_text("{ не json", encoding="utf-8")
    store = RingConfigStore(path)

    assert store.load() is None
    assert len(store.resolve(_profile_ring())) == len(_profile_ring())


def test_saved_file_is_readable_json(tmp_path):
    store = _store(tmp_path)
    store.save(["qwen3:4b"])

    assert json.loads((tmp_path / "ring.json").read_text(encoding="utf-8")) == {
        "models": ["qwen3:4b"]
    }


def test_embedding_model_is_recognized_by_configuration():
    assert is_embedding_model("nomic-embed-text:latest") is True


@pytest.mark.parametrize(
    "model", ["mxbai-embed-large", "bge-m3:latest", "all-minilm:l6-v2"]
)
def test_common_embedding_families_are_recognized(model):
    assert is_embedding_model(model) is True


@pytest.mark.parametrize("model", ["qwen3:4b", "llama3.1:8b", "gemma3:12b"])
def test_generative_models_are_not_mistaken_for_embeddings(model):
    assert is_embedding_model(model) is False


def test_embedding_model_saved_earlier_is_dropped_from_the_ring(tmp_path):
    store = _store(tmp_path)
    store.save(["qwen3:4b", "nomic-embed-text:latest"])

    assert [e.model for e in store.resolve(_profile_ring())] == ["qwen3:4b"]


def test_ring_of_embeddings_only_falls_back_to_the_profile(tmp_path):
    store = _store(tmp_path)
    store.save(["nomic-embed-text:latest"])

    assert [e.model for e in store.resolve(_profile_ring())] == [
        e.model for e in _profile_ring()
    ]


def test_model_deleted_from_the_runtime_drops_out_of_the_ring(tmp_path):
    """Удалённые веса не должны висеть в составе строкой «не установлена»."""
    store = _store(tmp_path)
    store.save(["qwen3:4b", "gemma3:4b"])

    resolved = store.resolve(_profile_ring(), installed={"qwen3:4b"})

    assert [e.model for e in resolved] == ["qwen3:4b"]


def test_reinstalled_model_returns_to_its_place_in_the_ring(tmp_path):
    """Состав в файле не переписывается: модель вернётся на свою позицию."""
    store = _store(tmp_path)
    store.save(["qwen3:4b", "gemma3:4b"])
    store.resolve(_profile_ring(), installed={"qwen3:4b"})

    resolved = store.resolve(_profile_ring(), installed={"qwen3:4b", "gemma3:4b"})

    assert [e.model for e in resolved] == ["qwen3:4b", "gemma3:4b"]
    assert store.load() == ["qwen3:4b", "gemma3:4b"]


def test_ring_falls_back_to_the_profile_when_nothing_is_installed(tmp_path):
    """Пустое кольцо не бывает: без установленных моделей остаётся план."""
    store = _store(tmp_path)
    store.save(["qwen3:4b"])

    resolved = store.resolve(_profile_ring(), installed=set())

    assert [e.model for e in resolved] == [e.model for e in _profile_ring()]


def test_unknown_installed_set_keeps_the_saved_composition(tmp_path):
    """Недоступный runtime — не повод вычищать сохранённый состав."""
    store = _store(tmp_path)
    store.save(["qwen3:4b", "gemma3:4b"])

    resolved = store.resolve(_profile_ring(), installed=None)

    assert [e.model for e in resolved] == ["qwen3:4b", "gemma3:4b"]


def test_profile_ring_also_drops_models_absent_from_the_runtime(tmp_path):
    """Кольцо по профилю — тоже кольцо: неустановленным в нём делать нечего."""
    resolved = _store(tmp_path).resolve(_profile_ring(), installed={"qwen3:4b"})

    assert [e.model for e in resolved] == ["qwen3:4b"]


def test_profile_ring_stays_whole_on_a_machine_without_models(tmp_path):
    """Свежая установка: состав читается как план загрузки, а не как пустота."""
    resolved = _store(tmp_path).resolve(_profile_ring(), installed=set())

    assert [e.model for e in resolved] == [e.model for e in _profile_ring()]
