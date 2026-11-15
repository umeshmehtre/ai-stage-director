# AI Stage Director

Voice-controlled 3D stage system. Speech → text (Whisper) → structured JSON (Qwen-2.5, LoRA fine-tuned) → speech (Kokoro).

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

## Fine-tuning (GPU required)
```bash
pip install -e ".[ml,train]"
python scripts/generate_dataset.py                       # data/train.jsonl, test.jsonl, handwritten_test.jsonl
python scripts/evaluate.py --model-id Qwen/Qwen2.5-1.5B-Instruct --data data/test.jsonl --label baseline_test
python scripts/finetune.py                               # LoRA, merged model -> models/qwen-stage-director
python scripts/evaluate.py --model-id models/qwen-stage-director --data data/test.jsonl --label finetuned_test
```
Serve the fine-tuned model:
```bash
STAGE_DIRECTOR_BACKEND=llm STAGE_LLM_MODEL_ID=models/qwen-stage-director uvicorn stage_director.main:app
```

## Results
| Model | Dataset | JSON valid | Preset accuracy |
|---|---|---|---|
| Qwen2.5-1.5B (base) | generated test (80) | _fill in_ | _fill in_ |
| Qwen2.5-1.5B + LoRA | generated test (80) | _fill in_ | _fill in_ |
| Qwen2.5-1.5B (base) | handwritten (10) | _fill in_ | _fill in_ |
| Qwen2.5-1.5B + LoRA | handwritten (10) | _fill in_ | _fill in_ |

Test commands use phrases held out from training to avoid leakage. The handwritten set is never used in training.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| STAGE_DIRECTOR_BACKEND | rules | `rules` or `llm` |
| STAGE_LLM_MODEL_ID | Qwen/Qwen2.5-1.5B-Instruct | Hugging Face id or local model path |
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
| GET | /health | Status, version, GPU availability, active backend |
| POST | /direct | `{"command": "..."}` → dialogue + lighting preset |
| POST | /transcribe | Audio file → transcript (Whisper) |
| POST | /direct-voice | Audio file → transcript + lighting + latency |

## Roadmap
- [x] M1: API service, tests, Docker
- [x] M2: Whisper + Qwen integration
- [x] M3: LoRA fine-tuning + evaluation
- [ ] M4: Kokoro TTS + 3D front end
- [ ] M5: MLflow tracking, Azure deployment