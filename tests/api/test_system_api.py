def test_health_returns_ok(client):
    response = client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info_exposes_local_model_configuration(client):
    payload = client.get("/system/info").json()
    assert payload["embedding_dim"] > 0
    assert payload["llm_model"]
    assert payload["embedding_model"]
