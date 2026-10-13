"""
Pydantic schemas for request/response validation and structured errors.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums expressed as Literal-style constrained strings
# ---------------------------------------------------------------------------

ScamCategory = str  # One of: FINANCIAL, JOB_SCAM, VISA_SCHOLARSHIP,
                    # TELEGRAM_ACCOUNT_TAKEOVER, CREDENTIAL_HARVEST,
                    # GENERAL_SCAM, BENIGN

RiskLevel = str     # LOW | MEDIUM | HIGH | CRITICAL


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class TextScanRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Message text to scan (Amharic, Afan Oromo, English, or mixed).",
        examples=[
            "እንኳን ደስ አለዎት! በቴሌብር ሎተሪ 50,000 ብር አሸንፈዋል። ሊንክ ይጫኑ።",
            "We are hiring! Work from home, earn 25,000 birr/week. Send CV to t.me/jobs_et",
        ],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty or whitespace only")
        return v.strip()


class URIScanRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=4,
        max_length=4096,
        description="URL to analyze.",
        examples=["http://telebirr-login.verify-account.xyz/secure"],
    )

    @field_validator("url")
    @classmethod
    def validate_url_shape(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", v):
            v = "http://" + v
        if not re.match(r"^https?://[^\s]+$", v, re.IGNORECASE):
            raise ValueError("Malformed URL")
        if len(v) > 4096:
            raise ValueError("URL too long")
        return v


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class Reason(BaseModel):
    code: str
    message: str
    weight: float


class CategoryScore(BaseModel):
    category: ScamCategory
    score: float = Field(..., ge=0.0)


class TextScanResponse(BaseModel):
    is_phishing: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    scam_category: str
    category_scores: List[CategoryScore] = []
    reasons: List[Reason]
    matched_keywords: List[str] = []
    language_detected: str = "unknown"


class URLScanResponse(BaseModel):
    is_phishing: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    scam_category: str
    url: str
    domain: str
    registered_domain: str
    tld: str
    impersonated_brand: Optional[str] = None
    brand_sector: Optional[str] = None
    reasons: List[Reason]
    indicators: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    supported_languages: List[str]
    supported_categories: List[str]