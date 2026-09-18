from app.core.config import Settings


def test_database_url_is_built_from_parts():
    settings = Settings(
        postgres_user="u", postgres_password="p", postgres_host="h",
        postgres_port=1234, postgres_db="db",
    )
    assert settings.database_url == "postgresql+psycopg://u:p@h:1234/db"


def test_default_reranker_is_enabled_for_russian_search():
    settings = Settings(secret_key="test-secret")

    assert settings.rerank_enabled is True
    assert settings.reranker_model == "DiTy/cross-encoder-russian-msmarco"
    assert settings.search_min_rerank_score == 0.2
