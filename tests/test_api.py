from fastapi.testclient import TestClient

from stage_director.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu_available"], bool)


def test_direct_red_command_sets_red_alert():
    response = client.post("/direct", json={"command": "Turn the lights RED"})
    assert response.status_code == 200
    assert response.json()["lighting_preset"] == "red_alert"


def test_direct_other_command_is_neutral():
    response = client.post("/direct", json={"command": "Hello stage"})
    assert response.status_code == 200
    assert response.json()["lighting_preset"] == "neutral"


def test_direct_rejects_empty_command():
    response = client.post("/direct", json={"command": "   "})
    assert response.status_code == 422