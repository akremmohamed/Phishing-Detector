"""
image_analyzer.py
=================

Image-based phishing / scam detection for the Ethiopian ecosystem.

Pipeline
--------
1.  Decode & validate the uploaded bytes (Pillow).
.  OCR the image (pytesseract if available; falls back to EasyOCR; else
    returns empty text without raising).
3.  Detect QR codes (pyzbar if available; falls back to OpenCV's
    QRCodeDetector; else skipped cleanly).
4.  Extract any URLs from the decoded QR payloads.
5.  Feed extracted text → TextAnalyzer and each URL → URLAnalyzer.
6.  Aggregate the individual risk scores into a single image-level score
    with a bounded, explainable weighting scheme.

The module never raises on OCR / QR failure — every dependency is loaded
lazily and defensively, so the caller always receives a valid
`ImageAnalysisResult`.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("image_analyzer")


# ============================================================================
# OPTIONAL DEPENDENCIES — loaded lazily, guarded, never fatal
# ============================================================================

# ---- OCR: pytesseract ----------------------------------------------------
try:
    import pytesseract  # type: ignore
    _PYTESSERACT_AVAILABLE = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    _PYTESSERACT_AVAILABLE = False


# ---- OCR: EasyOCR (fallback) --------------------------------------------
try:
    import easyocr  # type: ignore
    _EASYOCR_AVAILABLE = True
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore
    _EASYOCR_AVAILABLE = False


# ---- QR: pyzbar ----------------------------------------------------------
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode  # type: ignore
    _PYZBAR_AVAILABLE = True
except Exception:  # pragma: no cover
    _pyzbar_decode = None  # type: ignore
    _PYZBAR_AVAILABLE = False


# ---- QR: OpenCV ----------------------------------------------------------
try:
    import cv  # type: ignore
    import numpy as np  # type: ignore
    _CV_AVAILABLE = True
except Exception:  # pragma: no cover
    cv = None  # type: ignore
    np = None  # type: ignore
    _CV_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

#: Tesseract language packs. Amharic = "amh", English = "eng".
#: If "amh" is not installed, we retry with "eng" only.
TESSERACT_LANGS_PRIMARY = "amh+eng"
TESSERACT_LANGS_FALLBACK = "eng"

#: EasyOCR language list (fallback). EasyOCR does NOT ship Amharic;
#: English + Latin scripts are the practical fallback.
EASYOCR_LANGS = ["en"]

#: Max pixels we OCR at, to bound CPU time on huge uploads.
MAX_OCR_DIMENSION = 400

#: URL regex — captures http(s) URLs and scheme-less host.tld forms.
_RE_URL = re.compile(
    r"""(?xi)
    \b(
        (?:https?://|www\.)[^\s<>"'\)\]]+
        |
        (?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+
        (?:com|net|org|et|info|biz|xyz|top|site|online|link|live|tk|ml|ga|cf|gq|cc|click|shop|store|app)
        (?:/[^\s<>"'\)\]]*)?
    )\b
    """
)

#: "t.me/..."-style Telegram links should also be captured.
_RE_TELEGRAM = re.compile(
    r"""(?xi)\b(?:https?://)?(?:t\.me|telegram\.me)/[A-Za-z0-9_]{4,}\b"""
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OCRResult:
    text: str = ""
    engine: str = "none"
    languages: str = ""
    error: Optional[str] = None


@dataclass
class QRResult:
    payloads: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    engine: str = "none"
    error: Optional[str] = None


@dataclass
class ImageFinding:
    code: str
    message: str
    weight: float


@dataclass
class ImageAnalysisResult:
    filename: str
    content_type: str
    size_bytes: int
    extracted_text: str = ""
    detected_urls: List[str] = field(default_factory=list)
    qr_payloads: List[str] = field(default_factory=list)
    ocr_engine: str = "none"
    qr_engine: str = "none"
    text_analysis: Optional[Dict[str, Any]] = None
    url_analyses: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[ImageFinding] = field(default_factory=list)
    image_risk_score: float = 0.0
    is_phishing: bool = False
    risk_level: str = "LOW"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "extracted_text": self.extracted_text,
            "detected_urls": list(self.detected_urls),
            "qr_payloads": list(self.qr_payloads),
            "ocr_engine": self.ocr_engine,
            "qr_engine": self.qr_engine,
            "text_analysis": self.text_analysis,
            "url_analyses": self.url_analyses,
            "indicators": [f.code for f in self.findings],
            "reasons": [
                {"code": f.code, "message": f.message, "weight": round(f.weight, 3)}
                for f in self.findings
            ],
            "risk_score": self.image_risk_score,
            "is_phishing": self.is_phishing,
            "risk_level": self.risk_level,
            "error": self.error,
        }


# ============================================================================
# IMAGE LOADING
# ============================================================================

def _decode_image(data: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Decode bytes → PIL.Image. Returns (image, error_message).

    Handles:
      * empty input
      * corrupt / non-image data (UnidentifiedImageError)
      * EXIF orientation
      * RGB conversion
    """
    if not data:
        return None, "Empty image payload."

    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode to surface corruption now
    except UnidentifiedImageError:
        return None, "Unrecognised image format."
    except Exception as exc:
        return None, f"Failed to decode image: {exc}"

    try:
        img = ImageOps.exif_transpose(img) or img
    except Exception:
        pass  # non-fatal

    try:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        pass  # non-fatal

    return img, None


def _resize_for_ocr(img: Image.Image, max_dim: int = MAX_OCR_DIMENSION) -> Image.Image:
    """Downscale images that exceed `max_dim` on the long side."""
    try:
        w, h = img.size
        longest = max(w, h)
        if longest <= max_dim:
            return img
        scale = max_dim / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.LANCZOS)
    except Exception:
        return img


# ============================================================================
# OCR ENGINES
# ============================================================================

def _ocr_with_tesseract(img: Image.Image) -> OCRResult:
    """
    Run Tesseract with Amharic + English. Falls back to English-only
    if the Amharic language pack is not installed. Never raises.
    """
    if not _PYTESSERACT_AVAILABLE or pytesseract is None:
        return OCRResult(error="pytesseract not available")

    # Attempt Amharic+English first
    for langs in (TESSERACT_LANGS_PRIMARY, TESSERACT_LANGS_FALLBACK):
        try:
            text = pytesseract.image_to_string(img, lang=langs)
            if text and text.strip():
                return OCRResult(
                    text=text.strip(),
                    engine="pytesseract",
                    languages=langs,
                )
        except pytesseract.TesseractNotFoundError:
            return OCRResult(error="Tesseract binary not installed")
        except Exception as exc:
            # Likely missing language pack (TesseractError) — try next langs
            logger.debug("Tesseract langs=%s failed: %s", langs, exc)
            continue

    return OCRResult(engine="pytesseract", error="No text detected")


def _ocr_with_easyocr(img: Image.Image) -> OCRResult:
    """
    Fallback OCR via EasyOCR. The reader is cached at module scope the
    first time it's needed to avoid repeated model loads.
    """
    if not _EASYOCR_AVAILABLE or easyocr is None or np is None:
        return OCRResult(error="easyocr not available")

    global _EASYOCR_READER
    try:
        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(EASYOCR_LANGS, gpu=False, verbose=False)
    except Exception as exc:
        logger.exception("EasyOCR reader init failed")
        return OCRResult(error=f"EasyOCR init failed: {exc}")

    try:
        arr = np.array(img)
        lines = _EASYOCR_READER.readtext(arr, detail=0)
        text = "\n".join(line for line in lines if line)
        return OCRResult(
            text=text.strip(),
            engine="easyocr",
            languages="+".join(EASYOCR_LANGS),
        )
    except Exception as exc:
        logger.exception("EasyOCR inference failed")
        return OCRResult(error=f"EasyOCR inference failed: {exc}")


_EASYOCR_READER = None  # lazily initialised


def _extract_text(img: Image.Image) -> OCRResult:
    """
    Try Tesseract first; if unavailable/empty, try EasyOCR; else return
    an empty result with an error annotation.
    """
    result = _ocr_with_tesseract(img)
    if result.text:
        return result

    if _EASYOCR_AVAILABLE:
        easy_result = _ocr_with_easyocr(img)
        if easy_result.text:
            return easy_result
        if not result.error:
            result = easy_result

    if not result.error:
        result.error = "No OCR engine produced text"
    return result


# ============================================================================
# QR ENGINES
# ============================================================================

def _qr_with_pyzbar(img: Image.Image) -> QRResult:
    if not _PYZBAR_AVAILABLE or _pyzbar_decode is None:
        return QRResult(error="pyzbar not available")
    try:
        codes = _pyzbar_decode(img)
    except Exception as exc:
        logger.exception("pyzbar decode failed")
        return QRResult(error=f"pyzbar failed: {exc}")

    payloads: List[str] = []
    for code in codes or []:
        try:
            payload = code.data.decode("utf-8", errors="replace").strip()
            if payload:
                payloads.append(payload)
        except Exception:
            continue
    return QRResult(payloads=payloads, engine="pyzbar")


def _qr_with_opencv(img: Image.Image) -> QRResult:
    if not _CV_AVAILABLE or cv is None or np is None:
        return QRResult(error="opencv not available")
    try:
        arr = np.array(img.convert("RGB"))
        gray = cv.cvtColor(arr, cv.COLOR_RGBGRAY)
        detector = cv.QRCodeDetector()
        # Try multi-code API first (OpenCV ≥ 4.5)
        payloads: List[str] = []
        try:
            ok, decoded, _, _ = detector.detectAndDecodeMulti(gray)
            if ok and decoded:
                payloads.extend([d.strip() for d in decoded if d and d.strip()])
        except Exception:
            pass
        if not payloads:
            data, _, _ = detector.detectAndDecode(gray)
            if data and data.strip():
                payloads.append(data.strip())
        return QRResult(payloads=payloads, engine="opencv")
    except Exception as exc:
        logger.exception("OpenCV QR decode failed")
        return QRResult(error=f"opencv failed: {exc}")


def _extract_qr(img: Image.Image) -> QRResult:
    """
    Run pyzbar first (fast, robust for photographed receipts), fall back
    to OpenCV's QRCodeDetector. Merge both results — deduplicated.
    """
    primary = _qr_with_pyzbar(img)
    secondary = _qr_with_opencv(img)

    payloads = list(dict.fromkeys(primary.payloads + secondary.payloads))
    engine_parts = [
        r.engine for r in (primary, secondary) if r.payloads and r.engine != "none"
    ]
    engine = "+".join(dict.fromkeys(engine_parts)) if engine_parts else "none"

    error: Optional[str] = None
    if not payloads:
        errors = [r.error for r in (primary, secondary) if r.error]
        if errors:
            error = "; ".join(errors)

    return QRResult(payloads=payloads, engine=engine, error=error)


# ============================================================================
# URL EXTRACTION FROM TEXT / QR PAYLOADS
# ============================================================================

def _extract_urls(*sources: str) -> List[str]:
    """Find and normalise URLs across any number of text sources."""
    urls: List[str] = []
    for src in sources:
        if not src:
            continue
        try:
            for match in _RE_URL.finditer(src):
                urls.append(match.group(0).strip())
            for match in _RE_TELEGRAM.finditer(src):
                urls.append(match.group(0).strip())
        except Exception:
            logger.exception("URL regex failed on a source string")
            continue

    normalised: List[str] = []
    seen = set()
    for raw in urls:
        url = raw.rstrip(".,;:!?)]}'\"").strip()
        if not url:
            continue
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "http://" + url.lstrip("/")
        if url.lower() not in seen:
            seen.add(url.lower())
            normalised.append(url)
    return normalised


# ============================================================================
# SCORING
# ============================================================================

def _aggregate_score(
    text_result: Optional[Dict[str, Any]],
    url_results: List[Dict[str, Any]],
    qr_present: bool,
) -> Tuple[float, bool, str]:
    """
    Combine text-analysis and URL-analysis risks into one image score.

    Weighting
    ---------
      * Highest URL score      × 0.55
      * Text-analysis score    × 0.35
      * QR presence bonus      + 0.05 (only if QR pointed at a URL)
      * Extra URLs bump        + 0.03 per additional URL, capped at +0.09

    The final score is clamped to [0.0, 1.0] and mapped to LOW/MEDIUM/HIGH/CRITICAL.
    """
    text_score = 0.0
    if isinstance(text_result, dict):
        try:
            text_score = float(text_result.get("confidence", 0.0))
        except Exception:
            text_score = 0.0

    url_scores = []
    for u in url_results:
        try:
            url_scores.append(float(u.get("risk_score", 0.0)))
        except Exception:
            continue

    top_url = max(url_scores) if url_scores else 0.0

    score = 0.55 * top_url + 0.35 * text_score

    if qr_present and url_scores:
        score += 0.05
    if len(url_scores) > 1:
        score += min(0.09, 0.03 * (len(url_scores) - 1))

    score = max(0.0, min(1.0, round(score, 4)))

    if score >= 0.80:
        level = "CRITICAL"
    elif score >= 0.60:
        level = "HIGH"
    elif score >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    is_phishing = score >= 0.5
    return score, is_phishing, level


# ============================================================================
# PUBLIC ANALYZER
# ============================================================================

class ImageAnalyzer:
    """
    Orchestrates OCR, QR decoding, URL extraction, and score aggregation.

    Parameters
    ----------
    text_analyzer : TextAnalyzer
        Instance from `text_analyzer_.TextAnalyzer`.
    url_analyzer : URLAnalyzer
        Instance from `url_analyzer_.URLAnalyzer`.
    max_bytes : int
        Hard cap on input size (default 5 MiB).
    """

    def __init__(
        self,
        text_analyzer: Any,
        url_analyzer: Any,
        max_bytes: int = 5 * 104 * 104,
    ) -> None:
        self.text_analyzer = text_analyzer
        self.url_analyzer = url_analyzer
        self.max_bytes = max_bytes

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def analyze_image(
        self,
        data: bytes,
        filename: str = "upload",
        content_type: str = "application/octet-stream",
    ) -> ImageAnalysisResult:
        """
        Analyse image bytes. Never raises — errors are captured in the result.
        """
        result = ImageAnalysisResult(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data) if data else 0,
        )

        # -------- Size guard --------
        if not data:
            result.error = "Empty upload."
            result.findings.append(ImageFinding(
                code="EMPTY_UPLOAD",
                message="Uploaded file is empty.",
                weight=0.0,
            ))
            return result

        if len(data) > self.max_bytes:
            result.error = (
                f"File size {len(data)} bytes exceeds limit "
                f"of {self.max_bytes} bytes."
            )
            result.findings.append(ImageFinding(
                code="FILE_TOO_LARGE",
                message=result.error,
                weight=0.0,
            ))
            return result

        # -------- Decode --------
        img, decode_error = _decode_image(data)
        if img is None:
            result.error = decode_error or "Failed to decode image."
            result.findings.append(ImageFinding(
                code="INVALID_IMAGE",
                message=result.error,
                weight=0.0,
            ))
            return result

        # Downscale for OCR / QR robustness on very large images
        ocr_img = _resize_for_ocr(img)

        # -------- QR decoding (do this first — cheap, high-signal) --------
        qr = _extract_qr(ocr_img)
        result.qr_engine = qr.engine
        result.qr_payloads = qr.payloads

        # -------- OCR --------
        ocr = _extract_text(ocr_img)
        result.ocr_engine = ocr.engine
        result.extracted_text = ocr.text or ""

        if ocr.error:
            result.findings.append(ImageFinding(
                code="OCR_PARTIAL",
                message=f"OCR note: {ocr.error}",
                weight=0.0,
            ))
        if qr.error:
            result.findings.append(ImageFinding(
                code="QR_PARTIAL",
                message=f"QR note: {qr.error}",
                weight=0.0,
            ))

        # -------- URL extraction from both OCR text and QR payloads --------
        urls = _extract_urls(result.extracted_text, *qr.payloads)
        result.detected_urls = urls

        if qr.payloads:
            result.findings.append(ImageFinding(
                code="QR_CODE_PRESENT",
                message=(
                    f"Decoded {len(qr.payloads)} QR code payload(s) "
                    f"via {qr.engine}."
                ),
                weight=0.,
            ))
        if urls:
            result.findings.append(ImageFinding(
                code="URLS_EXTRACTED",
                message=f"Extracted {len(urls)} URL(s) from image content.",
                weight=0.3,
            ))

        # -------- Text analysis --------
        text_result_dict: Optional[Dict[str, Any]] = None
        if result.extracted_text.strip():
            try:
                text_result = self.text_analyzer.analyze(result.extracted_text)
                text_result_dict = text_result.to_dict()
                result.text_analysis = text_result_dict
                if text_result_dict.get("is_phishing"):
                    result.findings.append(ImageFinding(
                        code="TEXT_PHISHING",
                        message=(
                            "Extracted text classified as phishing "
                            f"({text_result_dict.get('scam_category')})."
                        ),
                        weight=float(text_result_dict.get("confidence", 0.0)),
                    ))
            except Exception as exc:
                logger.exception("TextAnalyzer failed on OCR output")
                result.findings.append(ImageFinding(
                    code="TEXT_ANALYSIS_ERROR",
                    message=f"Text analysis failed: {exc}",
                    weight=0.0,
                ))
        else:
            result.findings.append(ImageFinding(
                code="NO_TEXT_EXTRACTED",
                message="No readable text was extracted from the image.",
                weight=0.0,
            ))

        # -------- URL analysis --------
        url_result_dicts: List[Dict[str, Any]] = []
        for url in urls:
            try:
                url_dict = self.url_analyzer.analyze_url(url)
                url_result_dicts.append(url_dict)
                if url_dict.get("is_phishing"):
                    result.findings.append(ImageFinding(
                        code="URL_PHISHING",
                        message=(
                            f"URL '{url}' classified as phishing "
                            f"({url_dict.get('risk_level', 'HIGH')})."
                        ),
                        weight=float(url_dict.get("risk_score", 0.0)),
                    ))
            except Exception as exc:
                logger.exception("URLAnalyzer failed on %s", url)
                result.findings.append(ImageFinding(
                    code="URL_ANALYSIS_ERROR",
                    message=f"URL analysis failed for {url}: {exc}",
                    weight=0.0,
                ))
        result.url_analyses = url_result_dicts

        # -------- Aggregate --------
        score, is_phishing, level = _aggregate_score(
            text_result=text_result_dict,
            url_results=url_result_dicts,
            qr_present=bool(qr.payloads),
        )
        result.image_risk_score = score
        result.is_phishing = is_phishing
        result.risk_level = level

        return result