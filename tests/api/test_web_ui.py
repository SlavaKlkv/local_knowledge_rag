"""Веб-интерфейс отдаётся тем же приложением, что и API."""

from fastapi.testclient import TestClient


def test_index_served_at_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Local Knowledge RAG" in response.text


def test_static_assets_available(client: TestClient) -> None:
    for path in ("/ui/app.js", "/ui/styles.css", "/ui/logo.svg"):
        assert client.get(path).status_code == 200, path


def test_brand_uses_knowledge_ring_logo(client: TestClient) -> None:
    page = client.get("/").text

    assert 'href="/ui/logo.svg"' in page
    assert 'class="logo-ring"' in page
    assert 'class="logo-fragment"' in page
    assert 'class="logo-source"' in page
    assert 'class="logo-flow"' not in page


def test_ring_success_notice_auto_hides_but_error_notice_does_not(
    client: TestClient,
) -> None:
    script = client.get("/ui/app.js").text

    assert 'if (kind === "ok")' in script
    assert "ringNotice.remaining = 5000" in script
    assert 'addEventListener("mouseenter", pauseRingNoticeDismissal)' in script
    assert 'addEventListener("mouseleave", resumeRingNoticeDismissal)' in script


def test_ui_does_not_shadow_api(client: TestClient) -> None:
    """Монтирование статики не должно перехватывать маршруты API."""
    assert client.get("/system/health").status_code == 200


def test_index_hidden_from_openapi(client: TestClient) -> None:
    """Страница интерфейса — не эндпоинт API, в схеме ей не место."""
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_knowledge_base_can_be_renamed_inline(client: TestClient) -> None:
    """Переименование правится в строке списка и уходит в API как PATCH."""
    script = client.get("/ui/app.js").text

    assert 'rename.title = "Переименовать базу знаний"' in script
    assert 'method: "PATCH"' in script
    assert 'form.className = "kb-rename"' in script
    assert ".kb-rename" in client.get("/ui/styles.css").text
