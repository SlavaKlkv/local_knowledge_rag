"""Веб-интерфейс отдаётся тем же приложением, что и API."""

from fastapi.testclient import TestClient


def test_index_served_at_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Local Knowledge RAG" in response.text


def test_static_assets_available(client: TestClient) -> None:
    for path in ("/ui/app.js", "/ui/styles.css"):
        assert client.get(path).status_code == 200, path


def test_ui_does_not_shadow_api(client: TestClient) -> None:
    """Монтирование статики не должно перехватывать маршруты API."""
    assert client.get("/system/health").status_code == 200


def test_index_hidden_from_openapi(client: TestClient) -> None:
    """Страница интерфейса — не эндпоинт API, в схеме ей не место."""
    assert "/" not in client.get("/openapi.json").json()["paths"]
