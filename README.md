# AI Stage Director

Voice-controlled 3D stage system. Speech → text (Whisper) → structured JSON (Qwen-2.5) → speech (Kokoro).

## Architecture
Browser (3D stage) ⇄ Cloudflare tunnel ⇄ FastAPI on Colab GPU

```
audio → /direct-voice → Whisper → transcript → Director (rules | Qwen) → validated JSON
```

## Quickstart (local, rules backend)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
uvicorn stage_director.main:app --reload
```
Open http://localhost:8000/docs

## Run with real models (GPU machine / Colab)
```bash
pip install -e ".[ml]"
STAGE_DIRECTOR_BACKEND=llm uvicorn stage_director.main:app --host 0.0.0.0 --port 8000
```

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| STAGE_DIRECTOR_BACKEND | rules | `rules` or `llm` |
| STAGE_LLM_MODEL_ID | Qwen/Qwen2.5-1.5B-Instruct | Hugging Face model |
| STAGE_WHISPER_MODEL_SIZE | small | Whisper size |
| STAGE_MAX_UPLOAD_MB | 10 | Upload limit |
| STAGE_LOG_LEVEL | INFO | Logging level |

## Docker
```bash
docker build -t stage-director .
docker run -p 8000:8000 stage-director
```

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /health | Status, version, GPU availability |
| POST | /direct | `{"command": "..."}` → dialogue + lighting preset |
| POST | /transcribe | Audio file → transcript (Whisper) |
| POST | /direct-voice | Audio file → transcript + lighting + latency |

## Roadmap
- [x] M1: API service, tests, Docker
- [x] M2: Whisper + Qwen integration
- [ ] M3: Fine-tune Qwen (Unsloth)
- [ ] M4: Kokoro TTS + 3D front end
- [ ] M5: MLflow tracking, Azure deployment