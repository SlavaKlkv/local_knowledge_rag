from app.core.config import Settings


def test_database_url_is_built_from_parts():
    settings = Settings(
        postgres_user="u", postgres_password="p", postgres_host="h",
        postgres_port=1234, postgres_db="db",
    )
    assert settings.database_url == "postgresql+psycopg://u:p@h:1234/db"
