import base64

import pytest
from fastapi.testclient import TestClient

from stage_director.api.dependencies import get_synthesizer, get_transcriber
from stage_director.main import app


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: bytes) -> str:
        return self._text


class FakeSynthesizer:
    def synthesize(self, text: str) -> bytes:
        return b"RIFFfake"


class BrokenSynthesizer:
    def synthesize(self, text: str) -> bytes:
        raise RuntimeError("boom")


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_stage_text_returns_audio(client):
    app.dependency_overrides[get_synthesizer] = lambda: FakeSynthesizer()
    response = client.post("/stage-text", json={"command": "turn the lights red"})
    assert response.status_code == 200
    body = response.json()
    assert body["direction"]["lighting_preset"] == "red_alert"
    assert base64.b64decode(body["audio_base64"]) == b"RIFFfake"
    assert body["timings"]["total_ms"] >= 0


def test_stage_text_without_tts_has_no_audio(client):
    app.dependency_overrides[get_synthesizer] = lambda: None
    response = client.post("/stage-text", json={"command": "hello"})
    assert response.status_code == 200
    assert response.json()["audio_base64"] is None


def test_stage_text_survives_tts_failure(client):
    app.dependency_overrides[get_synthesizer] = lambda: BrokenSynthesizer()
    response = client.post("/stage-text", json={"command": "turn the lights red"})
    assert response.status_code == 200
    body = response.json()
    assert body["direction"]["lighting_preset"] == "red_alert"
    assert body["audio_base64"] is None


def test_stage_voice_full_pipeline(client):
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber("lights red")
    app.dependency_overrides[get_synthesizer] = lambda: FakeSynthesizer()
    response = client.post(
        "/stage", files={"file": ("clip.webm", b"fake-audio", "audio/webm")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "lights red"
    assert body["audio_base64"] is not None


def test_speak_returns_503_when_tts_disabled(client):
    app.dependency_overrides[get_synthesizer] = lambda: None
    response = client.post("/speak", json={"text": "hello"})
    assert response.status_code == 503