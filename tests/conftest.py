import pytest
from fastapi.testclient import TestClient

from app.main import create_app


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
