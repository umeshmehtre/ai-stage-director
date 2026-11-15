# AI Stage Director

Voice-controlled 3D stage. Speech → text (Whisper) → structured JSON (Qwen2.5, LoRA fine-tuned) → speech (Kokoro) → Three.js stage.

## Architecture
```
Browser (frontend/index.html) ⇄ Cloudflare tunnel ⇄ FastAPI on Colab GPU
mic → /stage → Whisper → Director (rules | Qwen) → Kokoro → {transcript, preset, audio, timings}
```

## Quickstart (local, rules backend)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
uvicorn stage_director.main:app --reload
```

## Run with real models (GPU machine / Colab)
```bash
pip install -e ".[ml,tts]"
apt-get install -y espeak-ng
pip install kokoro soundfile   # on Python 3.13 add: --ignore-requires-python
STAGE_DIRECTOR_BACKEND=llm STAGE_TTS_BACKEND=kokoro \
  uvicorn stage_director.main:app --host 0.0.0.0 --port 8000
```

## Frontend
```bash
cd frontend && python -m http.server 5500
```
Open http://localhost:5500, paste the API URL, then record or type a command.

## Fine-tuning (GPU required)
```bash
pip install -e ".[ml,train]"
python scripts/generate_dataset.py
python scripts/evaluate.py --model-id Qwen/Qwen2.5-1.5B-Instruct --data data/hard_test.jsonl --label baseline_hard
python scripts/finetune.py
python scripts/evaluate.py --model-id models/qwen-stage-director --data data/hard_test.jsonl --label finetuned_hard
```

## Results
| Model | Dataset | JSON valid | Preset accuracy |
|---|---|---|---|
| Base | generated test (80) | 97.5% | 95.0% |
| Fine-tuned | generated test (80) | _fill in_ | _fill in_ |
| Base | handwritten (10) | 100% | 100% |
| Fine-tuned | handwritten (10) | _fill in_ | _fill in_ |
| Base | hard set (30) | _fill in_ | _fill in_ |
| Fine-tuned | hard set (30) | _fill in_ | _fill in_ |

Test phrases are held out from training. The handwritten and hard sets are never used in training.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| STAGE_DIRECTOR_BACKEND | rules | `rules` or `llm` |
| STAGE_LLM_MODEL_ID | Qwen/Qwen2.5-1.5B-Instruct | HF id or local path |
| STAGE_WHISPER_MODEL_SIZE | small | Whisper size |
| STAGE_TTS_BACKEND | none | `none` or `kokoro` |
| STAGE_TTS_VOICE | af_heart | Kokoro voice |
| STAGE_MAX_UPLOAD_MB | 10 | Upload limit |
| STAGE_LOG_LEVEL | INFO | Logging level |

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /health | Status, GPU, active backends |
| POST | /direct | text → dialogue + lighting preset |
| POST | /transcribe | audio → transcript |
| POST | /direct-voice | audio → transcript + preset + latency |
| POST | /speak | text → WAV audio |
| POST | /stage | audio → transcript + preset + spoken reply + timings |
| POST | /stage-text | text → preset + spoken reply + timings |

## Docker
```bash
docker build -t stage-director .
docker run -p 8000:8000 stage-director
```

## Roadmap
- [x] M1: API service, tests, Docker
- [x] M2: Whisper + Qwen integration
- [x] M3: LoRA fine-tuning + evaluation
- [x] M4: Kokoro TTS + 3D front end
- [ ] M5: MLflow tracking, Azure deployment notes