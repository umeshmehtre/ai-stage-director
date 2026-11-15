"""HTTP endpoints. Thin layer: validate input, call a service, return output."""

import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from stage_director.api.dependencies import get_director, get_transcriber
from stage_director.config import Settings, get_settings
from stage_director.schemas import (
    DirectRequest,
    DirectResponse,
    HealthResponse,
    TranscribeResponse,
    VoiceDirectResponse,
)
from stage_director.services.director import Director
from stage_director.services.hardware import gpu_available
from stage_director.services.transcriber import Transcriber

logger = logging.getLogger(__name__)
router = APIRouter()


def _read_and_transcribe(
    file: UploadFile, transcriber: Transcriber, settings: Settings
) -> str:
    """Validate the upload, then convert it to text."""
    limit = settings.max_upload_mb * 1024 * 1024
    audio = file.file.read(limit + 1)

    if not audio:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(audio) > limit:
        raise HTTPException(
            status_code=413, detail=f"File larger than {settings.max_upload_mb} MB."
        )

    try:
        text = transcriber.transcribe(audio)
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=422, detail="Could not read this audio file.")

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected.")
    return text


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        gpu_available=gpu_available(),
    )


@router.post("/direct", response_model=DirectResponse, tags=["director"])
def direct(
    request: DirectRequest,
    director: Director = Depends(get_director),
) -> DirectResponse:
    return director.direct(request.command)


@router.post("/transcribe", response_model=TranscribeResponse, tags=["voice"])
def transcribe(
    file: UploadFile = File(...),
    transcriber: Transcriber = Depends(get_transcriber),
    settings: Settings = Depends(get_settings),
) -> TranscribeResponse:
    return TranscribeResponse(
        transcript=_read_and_transcribe(file, transcriber, settings)
    )


@router.post("/direct-voice", response_model=VoiceDirectResponse, tags=["voice"])
def direct_voice(
    file: UploadFile = File(...),
    transcriber: Transcriber = Depends(get_transcriber),
    director: Director = Depends(get_director),
    settings: Settings = Depends(get_settings),
) -> VoiceDirectResponse:
    started = time.perf_counter()
    text = _read_and_transcribe(file, transcriber, settings)
    direction = director.direct(text)
    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info("direct-voice finished in %d ms", latency_ms)
    return VoiceDirectResponse(
        transcript=text, direction=direction, latency_ms=latency_ms
    )