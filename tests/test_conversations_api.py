import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import create_app


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def client(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


@pytest.fixture
def knowledge_base_id(client):
    return client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]


def test_create_and_get_conversation(client, knowledge_base_id):
    created = client.post(
        "/conversations", json={"knowledge_base_id": knowledge_base_id, "title": "Отпуска"}
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    fetched = client.get(f"/conversations/{conversation_id}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Отпуска"


def test_creating_a_conversation_for_unknown_knowledge_base_is_rejected(client):
    response = client.post(
        "/conversations", json={"knowledge_base_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404


def test_unknown_conversation_returns_404(client):
    response = client.get(f"/conversations/{uuid.uuid4()}")
    assert response.status_code == 404


def test_messages_start_empty(client, knowledge_base_id):
    conversation_id = client.post(
        "/conversations", json={"knowledge_base_id": knowledge_base_id}
    ).json()["id"]

    response = client.get(f"/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    assert response.json() == []


def test_listing_messages_for_unknown_conversation_returns_404(client):
    response = client.get(f"/conversations/{uuid.uuid4()}/messages")
    assert response.status_code == 404
