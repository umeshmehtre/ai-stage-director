# AI Stage Director

Voice-controlled 3D stage system. Speech → text (Whisper) → structured JSON (Qwen-2.5) → speech (Kokoro).
This repo currently contains the FastAPI serving layer (Milestone 1).

## Architecture
Browser (3D stage) ⇄ Cloudflare tunnel ⇄ FastAPI on Colab GPU

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
uvicorn stage_director.main:app --reload
```
Open http://localhost:8000/docs

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

## Roadmap
- [x] M1: API service, tests, Docker
- [ ] M2: Whisper + Qwen integration
- [ ] M3: Fine-tune Qwen (Unsloth)
- [ ] M4: Kokoro TTS + 3D front end
- [ ] M5: MLflow tracking, Azure deployment