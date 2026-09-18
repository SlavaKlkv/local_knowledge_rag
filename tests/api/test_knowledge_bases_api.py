import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from tests.conftest import authenticate


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
    client = TestClient(app)
    authenticate(client)
    return client


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


def test_unknown_knowledge_base_is_indistinguishable_from_a_forbidden_one(client):
    """404 выдал бы факт существования чужой базы знаний, поэтому 403."""
    response = client.get(f"/knowledge-bases/{uuid.uuid4()}")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_deleting_a_knowledge_base(client):
    created = client.post("/knowledge-bases", json={"name": "Temp"})
    kb_id = created.json()["id"]

    response = client.delete(f"/knowledge-bases/{kb_id}")

    assert response.status_code == 204
    assert client.get(f"/knowledge-bases/{kb_id}").status_code == 403


def test_empty_name_is_rejected(client):
    response = client.post("/knowledge-bases", json={"name": ""})
    assert response.status_code == 422


def test_renaming_a_knowledge_base(client):
    kb_id = client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]

    response = client.patch(f"/knowledge-bases/{kb_id}", json={"name": "Кадры"})

    assert response.status_code == 200
    assert response.json()["name"] == "Кадры"
    assert client.get(f"/knowledge-bases/{kb_id}").json()["name"] == "Кадры"


def test_renaming_keeps_the_description(client):
    """Описание не присылали — значит, его не трогают."""
    kb_id = client.post(
        "/knowledge-bases", json={"name": "HR", "description": "Кадры"}
    ).json()["id"]

    updated = client.patch(f"/knowledge-bases/{kb_id}", json={"name": "People"}).json()

    assert updated["description"] == "Кадры"


def test_description_can_be_cleared_explicitly(client):
    kb_id = client.post(
        "/knowledge-bases", json={"name": "HR", "description": "Кадры"}
    ).json()["id"]

    updated = client.patch(
        f"/knowledge-bases/{kb_id}", json={"description": None}
    ).json()

    assert updated["description"] is None
    assert updated["name"] == "HR"


def test_renaming_to_an_empty_name_is_rejected(client):
    kb_id = client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]

    response = client.patch(f"/knowledge-bases/{kb_id}", json={"name": ""})

    assert response.status_code == 422


def test_update_without_fields_is_rejected(client):
    kb_id = client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]

    assert client.patch(f"/knowledge-bases/{kb_id}", json={}).status_code == 422


def test_renaming_someone_elses_knowledge_base_is_forbidden(client):
    response = client.patch(f"/knowledge-bases/{uuid.uuid4()}", json={"name": "X"})

    assert response.status_code == 403
