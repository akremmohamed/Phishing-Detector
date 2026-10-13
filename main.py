"""
main.py
=========

Secured FastAPI application for the Ethiopian Phishing & Scam Detection Engine.

Security features
-----------------
* **API Key authentication** via `X-API-Key` header (timing-safe comparison).
* **Rate limiting** with slowapi: 10 req/min for /scan-text & /scan-url,
  5 req/min for /scan-image. Optional Redis backend for multi-worker deployments.
* **CORS** driven by `ALLOWED_ORIGINS` env var (comma-separated).

Endpoints
---------
    GET  /api/v1/health          → public (no auth, no rate limit)
    POST /api/v1/scan-text       → auth + 10/min
    POST /api/v1/scan-url        → auth + 10/min
    POST /api/v1/scan-image      → auth + 5/min
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import (  # pyright: ignore[reportMissingImports]
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from image_analyzer import ImageAnalyzer
from text_analyzer import TextAnalyzer
from url_analyzer import URLAnalyzer


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("phishing_engine")


# ============================================================================
# CONFIG — environment-driven
# ============================================================================

#: Comma-separated list of allowed origins. Default is restrictive.
ALLOWED_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

#: API key expected in `X-API-Key`. MUST be set in production.
API_KEY: str = os.getenv("API_KEY", "")

#: Redis URI for slowapi. Falls back to in-memory when unset (single-worker only).
RATE_LIMIT_STORAGE_URI: str = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

#: Configurable per-endpoint limits.
RATE_TEXT: str = os.getenv("RATE_LIMIT_TEXT", "10/minute")
RATE_URL: str = os.getenv("RATE_LIMIT_URL", "10/minute")
RATE_IMAGE: str = os.getenv("RATE_LIMIT_IMAGE", "5/minute")

#: Image upload cap (bytes).
MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

ALLOWED_IMAGE_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/bmp", "image/tiff", "image/gif", "image/heic", "image/heif",
}
_READ_CHUNK = 64 * 1024


# ============================================================================
# RATE LIMITER  (slowapi)
# ============================================================================

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri=RATE_LIMIT_STORAGE_URI,
)

logger.info("Rate limit storage backend: %s", RATE_LIMIT_STORAGE_URI)


# ============================================================================
# API KEY AUTHENTICATION
# ============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """
    Dependency that validates the `X-API-Key` header against `API_KEY`.

    Uses `secrets.compare_digest` to prevent timing-based key recovery.
    Raises 401 if missing, 403 if invalid.
    """
    if not API_KEY:
        # Fail closed: if the server has no API_KEY configured, reject everything
        # (except public /health, which does not use this dependency).
        logger.error("API_KEY is not configured on the server — rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison — never use `==` on secrets
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


# ============================================================================
# APPLICATION STATE
# ============================================================================

class AppState:
    text_analyzer: TextAnalyzer | None = None
    url_analyzer: URLAnalyzer | None = None
    image_analyzer: ImageAnalyzer | None = None

    @property
    def ready(self) -> bool:
        return all(
            x is not None
            for x in (self.text_analyzer, self.url_analyzer, self.image_analyzer)
        )


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting detection engine …")
    t0 = time.perf_counter()
    state.url_analyzer = URLAnalyzer()
    state.text_analyzer = TextAnalyzer()
    state.image_analyzer = ImageAnalyzer(
        text_analyzer=state.text_analyzer,
        url_analyzer=state.url_analyzer,
        max_bytes=MAX_IMAGE_BYTES,
    )
    logger.info("Analyzers ready in %.2fs", time.perf_counter() - t0)
    yield
    logger.info("Shutting down.")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Ethiopian Phishing & Scam Detection Engine",
    description=(
        "AI-powered detection of phishing URLs, localized scam text "
        "(Amharic / Afan Oromo / English), and scam imagery."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS — driven by env
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request payload.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


# ============================================================================
# HELPERS
# ============================================================================

def _require_ready() -> None:
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine is still initializing, try again shortly.",
        )


async def _read_upload_capped(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        try:
            chunk = await upload.read(_READ_CHUNK)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read upload stream: {exc}",
            ) from exc
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Image exceeds maximum allowed size of "
                    f"{max_bytes // (1024 * 1024)} MB."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/api/v1/health", tags=["system"])
async def health() -> Dict[str, Any]:
    """Public health check — no auth, no rate limit."""
    return {
        "status": "ok",
        "version": app.version,
        "models_loaded": state.ready,
        "auth_configured": bool(API_KEY),
        "rate_limit_backend": RATE_LIMIT_STORAGE_URI.split("://", 1)[0],
        "supported_languages": ["amharic", "afan_oromo", "english", "mixed"],
        "max_image_bytes": MAX_IMAGE_BYTES,
    }


# ---------------------------------------------------------------------------
# TEXT SCAN
# ---------------------------------------------------------------------------

@app.post("/api/v1/scan-text", tags=["detection"])
@limiter.limit(RATE_TEXT)
async def scan_text(
    request: Request,
    payload: Dict[str, Any],
    _api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Body: { "text": "..." }
    Requires X-API-Key header. Rate limit: 10/min by default.
    """
    _require_ready()
    assert state.text_analyzer is not None

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'text' is required and must be a non-empty string.",
        )
    if len(text) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Field 'text' exceeds the 10,000-character limit.",
        )

    try:
        return state.text_analyzer.analyze_to_dict(text)
    except Exception as exc:
        logger.exception("scan_text failed")
        raise HTTPException(
            status_code=500, detail=f"Text analysis failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# URL SCAN
# ---------------------------------------------------------------------------

@app.post("/api/v1/scan-url", tags=["detection"])
@limiter.limit(RATE_URL)
async def scan_url(
    request: Request,
    payload: Dict[str, Any],
    _api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Body: { "url": "..." }
    Requires X-API-Key header. Rate limit: 10/min by default.
    """
    _require_ready()
    assert state.url_analyzer is not None

    url = payload.get("url", "")
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'url' is required and must be a non-empty string.",
        )
    url = url.strip()
    if len(url) > 4096:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="URL exceeds the 4,096-character limit.",
        )

    try:
        return await state.url_analyzer.analyze_url_async(url)
    except Exception as exc:
        logger.exception("scan_url failed")
        raise HTTPException(
            status_code=500, detail=f"URL analysis failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# IMAGE SCAN
# ---------------------------------------------------------------------------

@app.post("/api/v1/scan-image", tags=["detection"])
@limiter.limit(RATE_IMAGE)
async def scan_image(
    request: Request,
    file: UploadFile = File(...),
    _api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Multipart upload. Field name: `file`.
    Requires X-API-Key header. Rate limit: 5/min by default.
    """
    _require_ready()
    assert state.image_analyzer is not None

    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{content_type}'. "
                f"Please upload a JPEG, PNG, WEBP, BMP, TIFF, GIF, or HEIC image."
            ),
        )

    data = await _read_upload_capped(file, MAX_IMAGE_BYTES)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        analysis = state.image_analyzer.analyze_image(
            data=data,
            filename=file.filename or "upload",
            content_type=content_type or "application/octet-stream",
        )
    except Exception as exc:
        logger.exception("scan_image failed")
        raise HTTPException(
            status_code=500, detail=f"Image analysis failed: {exc}",
        ) from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass

    if analysis.error and analysis.error.startswith(("Unrecognised", "Failed to decode")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=analysis.error,
        )

    return analysis.to_dict()


# ============================================================================
# DEV ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        proxy_headers=True,  # respect X-Forwarded-For when behind a proxy
    )