# Full CPU image: voice + LLM + TTS. Slow-ish without a GPU but works anywhere.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# CPU-only PyTorch first (far smaller than the CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[ml,tts]" kokoro \
    && python -m spacy download en_core_web_sm

# Defaults can be overridden with environment variables at run time.
ENV STAGE_DIRECTOR_BACKEND=rules \
    STAGE_TTS_BACKEND=none \
    HF_HOME=/tmp/hf

RUN useradd -m -u 1000 user
USER user

EXPOSE 8000
CMD ["uvicorn", "stage_director.main:app", "--host", "0.0.0.0", "--port", "8000"]