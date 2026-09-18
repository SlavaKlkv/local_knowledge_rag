import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(autouse=True)
def no_real_model_downloads(monkeypatch):
    """Ни один тест не тянет веса из живого runtime'а.

    Загрузка модели — это гигабайты трафика и диска. Эндпоинт ставит её
    фоновой задачей, а TestClient выполняет такие задачи синхронно, поэтому
    забытая подмена run_download в одном тесте молча скачивала реальную
    модель на машину разработчика. Здесь этот путь закрыт: тест, которому
    нужен поток загрузки, подменяет httpx.stream сам — его подмена
    перекрывает эту.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Тест пытается скачать веса из настоящего runtime'а. "
            "Подмените ModelProvisioner.run_download или httpx.stream."
        )

    monkeypatch.setattr("app.llm.provisioning.httpx.stream", forbidden)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def register_user(client: TestClient, email: str = "user@example.com") -> dict:
    """Регистрирует пользователя и возвращает его данные с токеном."""
    created = client.post(
        "/auth/register", json={"email": email, "password": "password123"}
    )
    assert created.status_code == 201, created.text
    token = client.post(
        "/auth/login", json={"email": email, "password": "password123"}
    ).json()["access_token"]
    return {"id": created.json()["id"], "email": email, "token": token}


def authenticate(client: TestClient, email: str = "user@example.com") -> dict:
    """Регистрирует пользователя и проставляет его токен клиенту по умолчанию."""
    user = register_user(client, email)
    client.headers.update({"Authorization": f"Bearer {user['token']}"})
    return user
