"""HTTP endpoints. Thin layer: validate input, call a service, return output."""

from functools import lru_cache

from fastapi import APIRouter, Depends

from stage_director.config import Settings, get_settings
from stage_director.schemas import DirectRequest, DirectResponse, HealthResponse
from stage_director.services.director import RuleBasedDirector
from stage_director.services.hardware import gpu_available

router = APIRouter()


@lru_cache
def get_director() -> RuleBasedDirector:
    """Create the director once. Later, loading heavy models here is essential."""
    return RuleBasedDirector()


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
    director: RuleBasedDirector = Depends(get_director),
) -> DirectResponse:
    return director.direct(request.command)