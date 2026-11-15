import pytest
from fastapi.testclient import TestClient

from stage_director.api.dependencies import get_transcriber
from stage_director.main import app


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: bytes) -> str:
        return self._text


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def _audio():
    return {"file": ("clip.wav", b"fake-audio-bytes", "audio/wav")}


def test_direct_voice_happy_path(client):
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber(
        "turn the lights red"
    )
    response = client.post("/direct-voice", files=_audio())
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "turn the lights red"
    assert body["direction"]["lighting_preset"] == "red_alert"
    assert body["latency_ms"] >= 0


def test_direct_voice_no_speech(client):
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber("")
    response = client.post("/direct-voice", files=_audio())
    assert response.status_code == 422


def test_direct_voice_empty_file(client):
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber("hi")
    response = client.post(
        "/direct-voice", files={"file": ("clip.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400