# AI Stage Director

Voice-controlled 3D stage. Speech → text (Whisper) → structured JSON (Qwen2.5, LoRA fine-tuned) → speech (Kokoro) → Three.js stage.

```
Browser (frontend/index.html) ⇄ API ⇄ FastAPI
mic → /stage → Whisper → Director (rules | Qwen) → Kokoro → {transcript, preset, audio, timings}
```

## Run modes
| Mode | Needs | Voice in | LLM | Voice out |
|---|---|---|---|---|
| Offline (laptop) | Python only | optional (`.[asr]` + ffmpeg) | keyword rules | no |
| GPU (Colab) | T4 GPU | yes | fine-tuned Qwen | yes |
| CPU (Docker / HF Space) | 8 GB+ RAM | yes (slow) | fine-tuned Qwen (slow) | yes |

## Offline quickstart
```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn stage_director.main:app --port 8000
```
In a second terminal:
```bash
cd frontend && python -m http.server 5500
```
Open http://localhost:5500 and set the API URL to `http://localhost:8000`.

Optional voice input on a laptop: `pip install -e ".[asr]"`, install ffmpeg, and set `STAGE_WHISPER_MODEL_SIZE=base`.

## GPU quickstart (Colab T4)
```bash
pip install -e ".[ml,tts,gpu]"
apt-get install -y espeak-ng
# Kokoro's PyPI release caps Python < 3.13. On 3.13, install without dependency resolution:
pip install --no-deps --ignore-requires-python kokoro misaki
pip install loguru num2words addict espeakng-loader phonemizer-fork "spacy>=3.8,<4" soundfile
python -m spacy download en_core_web_sm
python scripts/colab_start.py --model models/qwen-stage-director
```
`colab_start.py` sets the NVIDIA library path Whisper needs, starts the server and prints a public URL.

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
| STAGE_WHISPER_MODEL_SIZE | small | `base` is faster on CPU |
| STAGE_WHISPER_LANGUAGE | en | `hi` for Hindi, empty = auto-detect |
| STAGE_WHISPER_TASK | transcribe | `translate` = speech to English text |
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
docker build -t stage-director .                      # slim, rules backend
docker build -f Dockerfile.full -t stage-director-full .   # CPU, voice + LLM + TTS
docker run -p 8000:8000 stage-director
```

## Known limitations
- Training data is synthetic and covers 4 lighting presets only.
- Free Colab gives a new public URL every session.
- CPU inference of the 1.5B model takes seconds per reply.
- Kokoro needs a manual install recipe on Python 3.13.

## Roadmap
- [x] M1 API, tests, Docker
- [x] M2 Whisper + Qwen
- [x] M3 LoRA fine-tuning + evaluation
- [x] M4 Kokoro TTS + 3D front end
- [ ] M5 MLflow tracking, monitoring, cloud deployment