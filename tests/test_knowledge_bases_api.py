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


def test_create_and_get_knowledge_base(client):
    created = client.post("/knowledge-bases", json={"name": "HR", "description": "Кадры"})
    assert created.status_code == 201
    kb_id = created.json()["id"]

    fetched = client.get(f"/knowledge-bases/{kb_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "HR"


def test_list_returns_created_knowledge_bases(client):
    client.post("/knowledge-bases", json={"name": "HR"})
    client.post("/knowledge-bases", json={"name": "Legal"})

    response = client.get("/knowledge-bases")

    assert {kb["name"] for kb in response.json()} == {"HR", "Legal"}


def test_unknown_knowledge_base_returns_404(client):
    response = client.get(f"/knowledge-bases/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_deleting_a_knowledge_base(client):
    created = client.post("/knowledge-bases", json={"name": "Temp"})
    kb_id = created.json()["id"]

    response = client.delete(f"/knowledge-bases/{kb_id}")

    assert response.status_code == 204
    assert client.get(f"/knowledge-bases/{kb_id}").status_code == 404


def test_empty_name_is_rejected(client):
    response = client.post("/knowledge-bases", json={"name": ""})
    assert response.status_code == 422
