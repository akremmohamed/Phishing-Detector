"""
url_analyzer.py
=================

Hardened URL & Domain Analyzer for the Ethiopian phishing detection engine.

Design goals
------------
* Zero unhandled exceptions — every public entry point returns a valid dict.
* Fast-path whitelisting for trusted domains (O(1) set membership).
* Async unshortening via httpx.AsyncClient with strict timeout / redirect caps.
* Pre-compiled regexes for all pattern-based checks.
* Deterministic, side-effect-free heuristics (safe to call concurrently).

Public API
----------
    analyzer = URLAnalyzer()
    result   = analyzer.analyze_url("http://telebirr-login.xyz")
    result   = await analyzer.analyze_url_async("http://bit.ly/abc")

Both `analyze_url` and `analyze_url_async` return a dict with the schema:

    {
        "url":            str,
        "risk_score":     float,        # 0.0 to 1.0
        "is_phishing":    bool,
        "indicators":     List[str],
        "domain_details": dict,
    }

Additional keys (reasons, resolved_url, was_shortened, whitelisted,
scam_category, impersonated_brand, brand_sector) are included for
richer consumers; they never break the required schema.
"""
from __future__ import annotations

import ipaddress
import logging
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, unquote

import httpx
import tldextract

logger = logging.getLogger("url_analyzer")


# ============================================================================
# CONFIGURATION
# ============================================================================

SUSPICIOUS_TLDS: Set[str] = {
    "top", "xyz", "tk", "ml", "ga", "cf", "gq", "cc", "click",
    "work", "link", "live", "online", "site", "website", "space",
    "press", "host", "loan", "download", "stream", "racing",
    "review", "country", "kim", "date", "men", "party", "trade",
    "science", "gdn", "bid", "icu", "cam", "rest", "bar", "beauty",
    "quest", "monster", "cyou", "shop", "store", "fun", "cool",
    "pro", "today", "world", "life", "info", "biz", "club",
}

SHORTENER_HOSTS: Set[str] = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy",
    "tiny.cc", "lnkd.in", "t.ly", "s.id", "shorte.st", "adf.ly",
    "bit.do", "mcaf.ee", "soo.gd", "q.gs", "po.st",
}

# ----------------------------------------------------------------------------
# TRUSTED DOMAIN WHITELIST
# Any host whose registered domain exactly equals (or ends with '.' +) an
# entry here is short-circuited as SAFE before any heuristics run.
# ----------------------------------------------------------------------------
WHITELISTED_DOMAINS: frozenset = frozenset({
    # --- Ethiopian financial ---
    "telebirr.et",
    "ethiotelecom.et",
    "combanketh.et",
    "cbe.com.et",
    "cbo.et",
    "bankofabyssinia.com",
    "dashenbanksc.com",
    "awashbank.com",
    "wegagen.com",
    "hibretbank.com",
    "zemenbank.com",
    "amoleapp.com",
    "amole.com.et",
    "safaricom.et",
    "m-pesa.safaricom.et",
    "ethswitch.com",
    "nibbank.com",
    "berhanbank.com.et",
    "bunna-bank.com",
    "abay-bank.com.et",
    "oromiabank.com",
    "tsehaybank.com.et",
    "siinqeebank.com",

    # --- Ethiopian government ---
    "gov.et",
    "ethiopia.gov.et",
    "ethiopost.et",
    "mols.gov.et",
    "moe.gov.et",
    "mofed.gov.et",
    "investethiopia.gov.et",
    "evisa.gov.et",
    "immigration.gov.et",
    "eta.gov.et",
    "moh.gov.et",
    "mofa.gov.et",

    # --- Global services commonly impersonated ---
    "google.com",
    "gmail.com",
    "youtube.com",
    "facebook.com",
    "fb.com",
    "instagram.com",
    "whatsapp.com",
    "wa.me",
    "telegram.org",
    "t.me",
    "telegram.me",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "netflix.com",
    "github.com",
    "cloudflare.com",
})

# ----------------------------------------------------------------------------
# BRAND REGISTRY
# ----------------------------------------------------------------------------
BRAND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "telebirr": {
        "sector": "bank",
        "canonical_domains": {"telebirr.et", "ethiotelecom.et"},
        "aliases": ["telebirr", "tele-birr", "tele_birr", "telebir", "telebirrbank"],
    },
    "cbe": {
        "sector": "bank",
        "canonical_domains": {"combanketh.et", "cbe.com.et"},
        "aliases": ["cbe", "commercialbank", "combank", "cbe-bank", "cbebirr"],
    },
    "cbo": {
        "sector": "bank",
        "canonical_domains": {"cbo.et"},
        "aliases": ["cbo", "coopbank", "cooperativebank", "cbo-bank"],
    },
    "boa": {
        "sector": "bank",
        "canonical_domains": {"bankofabyssinia.com"},
        "aliases": ["boa", "abyssinia", "bankofabyssinia", "abyssinia-bank"],
    },
    "dashen": {
        "sector": "bank",
        "canonical_domains": {"dashenbanksc.com"},
        "aliases": ["dashen", "dashenbank", "dashen-bank", "dashenbirr"],
    },
    "awash": {
        "sector": "bank",
        "canonical_domains": {"awashbank.com"},
        "aliases": ["awash", "awashbank", "awash-bank", "awashbirr"],
    },
    "wegagen": {
        "sector": "bank",
        "canonical_domains": {"wegagen.com"},
        "aliases": ["wegagen", "wegagenbank", "wegagen-bank"],
    },
    "hibret": {
        "sector": "bank",
        "canonical_domains": {"hibretbank.com"},
        "aliases": ["hibret", "hibretbank", "unitedbank", "hibret-bank"],
    },
    "zemen": {
        "sector": "bank",
        "canonical_domains": {"zemenbank.com"},
        "aliases": ["zemen", "zemenbank", "zemen-bank"],
    },
    "amole": {
        "sector": "wallet",
        "canonical_domains": {"amoleapp.com", "amole.com.et"},
        "aliases": ["amole", "amolewallet", "amole-app"],
    },
    "mpesa": {
        "sector": "wallet",
        "canonical_domains": {"safaricom.et", "m-pesa.safaricom.et"},
        "aliases": ["mpesa", "m-pesa", "mpesawallet", "safaricom"],
    },
    "ethswitch": {
        "sector": "wallet",
        "canonical_domains": {"ethswitch.com"},
        "aliases": ["ethswitch", "eth-switch", "ethswitchplc"],
    },
    "ethiotelecom": {
        "sector": "telecom",
        "canonical_domains": {"ethiotelecom.et"},
        "aliases": ["ethiotelecom", "ethio-telecom", "ethio_telecom", "etc"],
    },
    "ethio_gov": {
        "sector": "government",
        "canonical_domains": {
            "gov.et", "mols.gov.et", "moe.gov.et", "mofed.gov.et",
            "investethiopia.gov.et", "evisa.gov.et", "immigration.gov.et",
            "ethiopost.et",
        },
        "aliases": ["gov", "govet", "ethiogov", "mols", "moe", "mofed",
                    "evisa", "immigration", "investethiopia", "ethiopost"],
    },
    "telegram": {
        "sector": "telegram",
        "canonical_domains": {"telegram.org", "t.me", "telegram.me"},
        "aliases": ["telegram", "tme", "telegramme", "telegramapp"],
    },
    "facebook": {
        "sector": "platform",
        "canonical_domains": {"facebook.com", "fb.com", "m.facebook.com"},
        "aliases": ["facebook", "fb", "meta"],
    },
    "google": {
        "sector": "platform",
        "canonical_domains": {"google.com", "gmail.com", "youtube.com"},
        "aliases": ["google", "gmail", "youtube"],
    },
    "tiktok": {
        "sector": "platform",
        "canonical_domains": {"tiktok.com"},
        "aliases": ["tiktok", "tik-tok"],
    },
    "whatsapp": {
        "sector": "platform",
        "canonical_domains": {"whatsapp.com", "wa.me"},
        "aliases": ["whatsapp", "whats-app", "wa"],
    },
}

SECURITY_TOKENS: Set[str] = {
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "validate", "auth",
    "authenticate", "password", "passwd", "pwd", "otp", "pin",
    "token", "recover", "reset", "unlock", "billing", "wallet",
    "payment", "kyc", "idverify", "id-verify", "banking",
    "onlinebanking", "verifyaccount",
}

_HOMOGLYPH_TABLE = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t",
    "$": "s", "@": "a", "!": "i",
})

# ----------------------------------------------------------------------------
# PRE-COMPILED REGEXES  (compiled once at import time)
# ----------------------------------------------------------------------------
_RE_DOUBLE_LURE_TOKEN = re.compile(
    r"(login|verify|secure|account|update|confirm)[.\-]?"
    r"(login|verify|secure|account|update|confirm)",
    re.IGNORECASE,
)
_RE_BRAND_TLD_SANDWICH = re.compile(
    r"(paypal|apple|google|microsoft|amazon|netflix|bank|telebirr|cbe|dashen|awash)"
    r"[.\-](com|net|org|info|xyz|top|site)",
    re.IGNORECASE,
)
_RE_DOUBLE_HYPHENS = re.compile(r"--+")
_RE_IPV4_IN_PATH = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_RE_SCAM_LURE_KEYWORDS = re.compile(
    r"(free|gift|prize|winner|lottery|bonus|claim|reward)", re.IGNORECASE,
)
_RE_LEGACY_SCRIPT_EXT = re.compile(r"\.(php|asp|aspx|jsp|cgi)$", re.IGNORECASE)
_RE_TG_SUSPICIOUS_HANDLE = re.compile(
    r"(official|verify|verification|support|help|admin|service|"
    r"bank|wallet|airdrop|grant|bonus|reward|giveaway)",
    re.IGNORECASE,
)

# Tuples: (compiled_regex, code, message, weight)
SUSPICIOUS_PATTERNS: List[Tuple[re.Pattern, str, str, float]] = [
    (_RE_DOUBLE_LURE_TOKEN, "DOUBLE_LURE_TOKEN",
     "Duplicate security-lure tokens in URL — classic phishing pattern.", 1.0),
    (_RE_BRAND_TLD_SANDWICH, "BRAND_TLD_SANDWICH",
     "Brand name followed by a TLD-like token, then another TLD — spoofing pattern.", 1.5),
    (_RE_DOUBLE_HYPHENS, "DOUBLE_HYPHENS",
     "Host contains consecutive hyphens — common in lookalike domains.", 0.7),
    (_RE_IPV4_IN_PATH, "IP_IN_PATH",
     "URL contains an embedded IP address in path/query — obfuscation.", 1.0),
    (_RE_SCAM_LURE_KEYWORDS, "SCAM_LURE_KEYWORD",
     "URL contains a scam-lure keyword (free/prize/lottery/claim).", 0.8),
    (_RE_LEGACY_SCRIPT_EXT, "LEGACY_SCRIPT_EXT",
     "URL ends with a legacy script extension — often used in phishing kits.", 0.4),
]


# ============================================================================
# MODULE-LEVEL HELPERS
# ============================================================================

_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _normalize_homoglyphs(s: str) -> str:
    return s.lower().translate(_HOMOGLYPH_TABLE)


def _tokenize_host(host: str) -> List[str]:
    return [t for t in re.split(r"[.\-_0-9]+", host.lower()) if t]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _safe_urlparse(url: str):
    """urlparse wrapper that never raises on malformed input."""
    try:
        return urlparse(url)
    except Exception:
        return urlparse("")


def _safe_tld_extract(url: str):
    """tldextract wrapper that never raises."""
    try:
        ext = _TLD_EXTRACT(url)
        return (
            (ext.registered_domain or "").lower(),
            (ext.suffix or "").lower(),
        )
    except Exception:
        return "", ""


def _is_whitelisted(registered: str, host: str) -> bool:
    """
    Return True if the registered domain OR host is a trusted domain,
    or is a subdomain of one.
    """
    for candidate in (registered, host):
        if not candidate:
            continue
        if candidate in WHITELISTED_DOMAINS:
            return True
        # Subdomain check: "mail.google.com" endswith ".google.com"
        for trusted in WHITELISTED_DOMAINS:
            if candidate.endswith("." + trusted):
                return True
    return False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class URLFinding:
    code: str
    message: str
    weight: float


@dataclass
class URLAnalysisResult:
    url: str
    domain: str = ""
    registered_domain: str = ""
    tld: str = ""
    impersonated_brand: Optional[str] = None
    brand_sector: Optional[str] = None
    findings: List[URLFinding] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    resolved_url: Optional[str] = None
    was_shortened: bool = False
    whitelisted: bool = False
    parse_error: Optional[str] = None

    @property
    def raw_score(self) -> float:
        return sum(f.weight for f in self.findings)

    @property
    def risk_score(self) -> float:
        """Normalized 0.0–1.0 risk score."""
        if self.whitelisted:
            return 0.0
        # Saturating curve: raw 0 → 0.0, raw 2 → ~0.67, raw 5 → ~0.94
        return round(min(1.0, 1 - math.exp(-0.55 * self.raw_score)), 4)

    @property
    def is_phishing(self) -> bool:
        if self.whitelisted:
            return False
        return self.raw_score >= 2.0

    @property
    def risk_level(self) -> str:
        s = self.raw_score
        if s >= 5.0:
            return "CRITICAL"
        if s >= 3.0:
            return "HIGH"
        if s >= 1.2:
            return "MEDIUM"
        return "LOW"

    @property
    def scam_category(self) -> str:
        if self.whitelisted:
            return "BENIGN"
        if self.brand_sector in ("bank", "wallet", "telecom"):
            return "FINANCIAL"
        if self.brand_sector == "telegram":
            return "TELEGRAM_ACCOUNT_TAKEOVER"
        if self.brand_sector == "government":
            return "VISA_SCHOLARSHIP"
        if any(f.code in ("SECURITY_TOKENS", "PUNYCODE", "AT_SYMBOL")
               for f in self.findings):
            return "CREDENTIAL_HARVEST"
        if self.is_phishing:
            return "GENERAL_SCAM"
        return "BENIGN"

    def to_dict(self) -> Dict[str, Any]:
        """Return the required schema plus supplementary metadata."""
        # ---------- Required schema ----------
        payload: Dict[str, Any] = {
            "url": self.url,
            "risk_score": self.risk_score,
            "is_phishing": self.is_phishing,
            "indicators": [f.code for f in self.findings],
            "domain_details": {
                "host": self.domain,
                "registered_domain": self.registered_domain,
                "tld": self.tld,
                "subdomain_depth": self.indicators.get("subdomain_depth", 0),
                "host_length": self.indicators.get("host_length", 0),
                "entropy": self.indicators.get("host_entropy", 0.0),
                "has_punycode": self.indicators.get("has_punycode", False),
                "is_ip_host": self.indicators.get("is_ip_host", False),
                "uses_https": self.indicators.get("uses_https", False),
            },
            # ---------- Supplementary metadata ----------
            "risk_level": self.risk_level,
            "raw_score": round(self.raw_score, 3),
            "scam_category": self.scam_category,
            "impersonated_brand": self.impersonated_brand,
            "brand_sector": self.brand_sector,
            "whitelisted": self.whitelisted,
            "was_shortened": self.was_shortened,
            "resolved_url": self.resolved_url,
            "parse_error": self.parse_error,
            "reasons": [
                {
                    "code": f.code,
                    "message": f.message,
                    "weight": round(f.weight, 3),
                }
                for f in self.findings
            ],
            "indicators_detail": dict(self.indicators),
        }
        return payload


# ============================================================================
# ANALYZER
# ============================================================================

class URLAnalyzer:
    """
    Hardened, concurrent-safe URL analyzer.

    Parameters
    ----------
    similarity_threshold : float
        Levenshtein ratio above which a domain is flagged as a typosquat.
    entropy_threshold : float
        Shannon entropy (bits/char) above which a host label is flagged.
    redirect_timeout : float
        Seconds for HTTP redirect resolution.
    max_redirects : int
        Hard cap on redirect chain length.
    """

    BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        entropy_threshold: float = 3.8,
        redirect_timeout: float = 3.0,
        max_redirects: int = 5,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.entropy_threshold = entropy_threshold
        self.redirect_timeout = redirect_timeout
        self.max_redirects = max_redirects

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Synchronous analysis. Never raises.
        Always returns a dict matching the required schema.
        """
        try:
            result = self._analyze(url, resolved_url=None, was_shortened=False)
        except Exception as exc:  # pragma: no cover — defensive net
            logger.exception("analyze_url failed unexpectedly")
            result = self._error_result(url, f"{type(exc).__name__}: {exc}")
        return result.to_dict()

    async def analyze_url_async(self, url: str) -> Dict[str, Any]:
        """
        Async analysis with live redirect resolution for shorteners.
        Never raises. Always returns a dict matching the required schema.
        """
        resolved_url: Optional[str] = None
        was_shortened = False

        # ----- Step 1: quick pre-parse to detect shortener -----
        try:
            parsed = _safe_urlparse(url)
            host = (parsed.hostname or "").lower()
            registered, _ = _safe_tld_extract(url)

            # Whitelist short-circuits before any network call
            if _is_whitelisted(registered, host):
                return self._analyze(
                    url, resolved_url=None, was_shortened=False,
                ).to_dict()

            is_shortener = (
                host in SHORTENER_HOSTS
                or registered in SHORTENER_HOSTS
                or any(host.endswith("." + s) for s in SHORTENER_HOSTS)
            )
            if is_shortener:
                was_shortened = True
                resolved_url = await self._resolve_redirects(url)
        except Exception:
            logger.exception("Pre-parse / resolution failed; continuing offline")

        # ----- Step 2: full heuristic analysis -----
        try:
            result = self._analyze(
                url, resolved_url=resolved_url, was_shortened=was_shortened,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("analyze_url_async failed unexpectedly")
            result = self._error_result(url, f"{type(exc).__name__}: {exc}")
        return result.to_dict()

    # ------------------------------------------------------------------ #
    # ASYNC REDIRECT RESOLUTION
    # ------------------------------------------------------------------ #

    async def _resolve_redirects(self, url: str) -> Optional[str]:
        """
        Follow up to `max_redirects` HTTP redirects to reveal the final URL.

        Returns the final URL string on success, or None on any failure
        (network error, timeout, too many redirects, invalid URL, non-2xx).
        This method never raises.
        """
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=self.max_redirects,
                timeout=httpx.Timeout(self.redirect_timeout),
                headers={
                    "User-Agent": self.BROWSER_UA,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                verify=False,  # do not fail on bad certs during resolution
            ) as client:
                resp = await client.get(url)
                final = str(resp.url)
                if final and final != url:
                    return final
                return None
        except httpx.TooManyRedirects:
            logger.info("Too many redirects resolving %s", url)
            return None
        except httpx.TimeoutException:
            logger.info("Timeout resolving %s", url)
            return None
        except (httpx.HTTPError, httpx.InvalidURL):
            logger.info("HTTP error resolving %s", url)
            return None
        except Exception:
            logger.exception("Unexpected error resolving %s", url)
            return None

    # ------------------------------------------------------------------ #
    # CORE PIPELINE
    # ------------------------------------------------------------------ #

    def _analyze(
        self,
        url: str,
        resolved_url: Optional[str],
        was_shortened: bool,
    ) -> URLAnalysisResult:
        """
        Full synchronous pipeline. Returns a URLAnalysisResult.
        Any internal exception is captured into the result.
        """
        # -------- Parse (defensive) --------
        try:
            parsed = _safe_urlparse(url)
            host = (parsed.hostname or "").lower()
            registered, tld = _safe_tld_extract(url)
            if not registered:
                registered = host
        except Exception as exc:
            return self._error_result(url, f"Parse error: {exc}")

        result = URLAnalysisResult(
            url=url,
            domain=host,
            registered_domain=registered,
            tld=tld,
            resolved_url=resolved_url,
            was_shortened=was_shortened,
        )

        # -------- Whitelist fast-path --------
        if _is_whitelisted(registered, host):
            result.whitelisted = True
            result.findings.append(URLFinding(
                code="WHITELISTED",
                message=f"Domain '{registered or host}' is on the trusted whitelist.",
                weight=0.0,
            ))
            result.indicators = self._build_indicators(
                url, host, registered, tld, parsed, resolved_url, was_shortened,
            )
            return result

        # -------- Empty host guard --------
        if not host:
            result.findings.append(URLFinding(
                code="NO_HOST",
                message="URL contains no resolvable hostname.",
                weight=1.0,
            ))
            result.indicators = {"host": "", "raw_score": 1.0}
            return result

        # -------- Run all heuristic checks --------
        self._check_shortener(host, result)
        self._check_ip_host(host, result)
        self._check_tld(tld, result)
        self._check_length_structure(url, host, parsed, result)
        self._check_hyphens_digits(host, result)
        self._check_suspicious_tokens(host, parsed, result)
        self._check_impersonation(host, registered, result)
        self._check_port_scheme(parsed, result)
        self._check_punycode(host, result)
        self._check_encoded_payload(parsed, result)
        self._check_telegram_impersonation(host, parsed, result)
        self._check_entropy(host, result)
        self._check_suspicious_patterns(url, result)

        # -------- Re-analyze resolved target (if any) --------
        if resolved_url and resolved_url != url:
            self._analyze_resolved_target(resolved_url, result)

        # -------- Build indicators --------
        result.indicators = self._build_indicators(
            url, host, registered, tld, parsed, resolved_url, was_shortened,
        )
        return result

    def _analyze_resolved_target(
        self, resolved_url: str, parent: URLAnalysisResult,
    ) -> None:
        """Analyze the destination URL of a shortened link."""
        try:
            r_parsed = _safe_urlparse(resolved_url)
            r_host = (r_parsed.hostname or "").lower()
            r_registered, r_tld = _safe_tld_extract(resolved_url)
            if not r_registered:
                r_registered = r_host

            # If the destination is whitelisted, drop the SHORTENER penalty
            if _is_whitelisted(r_registered, r_host):
                parent.findings = [
                    f for f in parent.findings if f.code != "SHORTENER"
                ]
                parent.findings.append(URLFinding(
                    code="RESOLVED_TO_WHITELIST",
                    message=(
                        f"Shortener resolved to trusted domain "
                        f"'{r_registered}'."
                    ),
                    weight=0.0,
                ))
                return

            # Otherwise run additional checks against the destination
            self._check_tld(r_tld, parent)
            self._check_impersonation(r_host, r_registered, parent)
            self._check_suspicious_tokens(r_host, r_parsed, parent)
            self._check_suspicious_patterns(resolved_url, parent)
            self._check_entropy(r_host, parent)
        except Exception:
            logger.exception("Failed to analyze resolved target")

    def _build_indicators(
        self, url: str, host: str, registered: str, tld: str,
        parsed, resolved_url: Optional[str], was_shortened: bool,
    ) -> Dict[str, Any]:
        """Compute the raw indicator dict exposed on the result."""
        try:
            port = parsed.port
        except Exception:
            port = None
        try:
            return {
                "host": host,
                "path": parsed.path,
                "query": parsed.query,
                "scheme": parsed.scheme,
                "port": port,
                "is_shortener": (
                    host in SHORTENER_HOSTS
                    or registered in SHORTENER_HOSTS
                    or any(host.endswith("." + s) for s in SHORTENER_HOSTS)
                ),
                "url_length": len(url),
                "host_length": len(host),
                "subdomain_depth": self._subdomain_depth(host, registered),
                "num_digits_in_host": sum(c.isdigit() for c in host),
                "num_hyphens_in_host": host.count("-"),
                "has_punycode": "xn--" in host,
                "uses_https": parsed.scheme.lower() == "https",
                "is_ip_host": _is_ip_host(host),
                "host_entropy": round(_shannon_entropy(host), 3),
                "was_shortened": was_shortened,
                "resolved_url": resolved_url,
            }
        except Exception:
            return {"host": host, "url_length": len(url)}

    def _error_result(self, url: str, message: str) -> URLAnalysisResult:
        result = URLAnalysisResult(url=url, parse_error=message)
        result.findings.append(URLFinding(
            code="PARSE_ERROR",
            message=message,
            weight=1.0,
        ))
        result.indicators = {"parse_error": True, "raw_score": 1.0}
        return result

    # ------------------------------------------------------------------ #
    # DETECTION CHECKS
    # ------------------------------------------------------------------ #

    def _check_shortener(self, host: str, r: URLAnalysisResult) -> None:
        if host in SHORTENER_HOSTS or any(
            host.endswith("." + s) for s in SHORTENER_HOSTS
        ):
            r.findings.append(URLFinding(
                code="SHORTENER",
                message=f"URL uses link shortener '{host}' — destination hidden.",
                weight=1.2,
            ))

    def _check_ip_host(self, host: str, r: URLAnalysisResult) -> None:
        if _is_ip_host(host):
            r.findings.append(URLFinding(
                code="IP_HOST",
                message="URL uses a raw IP address instead of a domain name.",
                weight=2.0,
            ))

    def _check_tld(self, tld: str, r: URLAnalysisResult) -> None:
        if not tld:
            return
        if tld in SUSPICIOUS_TLDS:
            r.findings.append(URLFinding(
                code="SUSPICIOUS_TLD",
                message=(
                    f"High-risk top-level domain '.{tld}' "
                    f"frequently abused in phishing."
                ),
                weight=1.5,
            ))

    def _check_length_structure(
        self, url: str, host: str, parsed, r: URLAnalysisResult,
    ) -> None:
        if len(url) > 90:
            r.findings.append(URLFinding(
                code="LONG_URL",
                message=f"URL is unusually long ({len(url)} chars).",
                weight=0.8,
            ))
        if len(host) > 40:
            r.findings.append(URLFinding(
                code="LONG_HOST",
                message=f"Hostname is unusually long ({len(host)} chars).",
                weight=0.5,
            ))
        if host.count(".") >= 4:
            r.findings.append(URLFinding(
                code="DEEP_SUBDOMAIN",
                message="Host contains an excessive number of subdomain levels.",
                weight=0.6,
            ))
        if "@" in (parsed.netloc or ""):
            r.findings.append(URLFinding(
                code="AT_SYMBOL",
                message=(
                    "URL contains '@' in authority — commonly used to "
                    "obscure the real host."
                ),
                weight=1.5,
            ))

    def _check_hyphens_digits(self, host: str, r: URLAnalysisResult) -> None:
        if host.count("-") >= 3:
            r.findings.append(URLFinding(
                code="EXCESS_HYPHENS",
                message=(
                    f"Host contains {host.count('-')} hyphens — "
                    f"typosquatting pattern."
                ),
                weight=0.8,
            ))
        digits = sum(c.isdigit() for c in host)
        if digits >= 4 and digits / max(len(host), 1) > 0.18:
            r.findings.append(URLFinding(
                code="DIGIT_HEAVY_HOST",
                message="Host has an unusually high ratio of digits.",
                weight=0.6,
            ))

    def _check_suspicious_tokens(self, host: str, parsed, r: URLAnalysisResult) -> None:
        combined = f"{host}{parsed.path}".lower()
        hits = {tok for tok in SECURITY_TOKENS if tok in combined}
        if hits:
            r.findings.append(URLFinding(
                code="SECURITY_TOKENS",
                message=(
                    f"Contains credential/security lure tokens: "
                    f"{', '.join(sorted(hits)[:6])}."
                ),
                weight=0.4 + 0.15 * min(len(hits), 4),
            ))

    def _check_impersonation(
        self, host: str, registered: str, r: URLAnalysisResult,
    ) -> None:
        if not registered:
            return

        tokens = _tokenize_host(host)
        normalized_tokens = [_normalize_homoglyphs(t) for t in tokens]
        full_label = registered.split(".")[0] if "." in registered else registered
        full_label_norm = _normalize_homoglyphs(full_label)

        for brand, info in BRAND_REGISTRY.items():
            canonical: Set[str] = info["canonical_domains"]
            aliases: List[str] = info["aliases"]
            sector: str = info["sector"]

            # Skip if domain is already canonical
            if registered in canonical or any(
                registered.endswith("." + c) for c in canonical
            ):
                continue

            # Direct brand-token presence on a non-canonical host
            for alias in aliases:
                if alias in tokens or alias in normalized_tokens or alias in host:
                    r.impersonated_brand = r.impersonated_brand or brand
                    r.brand_sector = r.brand_sector or sector
                    r.findings.append(URLFinding(
                        code="BRAND_IN_NON_CANONICAL_HOST",
                        message=(
                            f"References brand '{brand}' ({sector}) in hostname, "
                            f"but registered domain '{registered}' is not an "
                            f"official {brand} domain."
                        ),
                        weight=2.5,
                    ))
                    return

            # Fuzzy typosquatting
            candidates = {full_label_norm} | {
                _normalize_homoglyphs(c) for c in aliases
            }
            best_ratio, best_target = 0.0, ""
            for cand in candidates:
                for alias in aliases + [full_label_norm]:
                    ratio = _ratio(
                        _normalize_homoglyphs(cand),
                        _normalize_homoglyphs(alias),
                    )
                    if ratio > best_ratio:
                        best_ratio, best_target = ratio, alias

            if self.similarity_threshold <= best_ratio < 1.0:
                r.impersonated_brand = r.impersonated_brand or brand
                r.brand_sector = r.brand_sector or sector
                r.findings.append(URLFinding(
                    code="TYPOSQUAT",
                    message=(
                        f"Registered domain '{registered}' closely resembles "
                        f"'{best_target}' ({brand}) — similarity {best_ratio:.2f}."
                    ),
                    weight=2.2 + (best_ratio - self.similarity_threshold),
                ))
                return

    def _check_port_scheme(self, parsed, r: URLAnalysisResult) -> None:
        if parsed.scheme and parsed.scheme.lower() == "http":
            r.findings.append(URLFinding(
                code="NO_TLS",
                message="Uses plain HTTP — legitimate services enforce HTTPS.",
                weight=1.0,
            ))
        try:
            port = parsed.port
        except Exception:
            port = None
        if port and port not in (80, 443):
            r.findings.append(URLFinding(
                code="ODD_PORT",
                message=f"Uses non-standard port {port}.",
                weight=0.7,
            ))

    def _check_punycode(self, host: str, r: URLAnalysisResult) -> None:
        if "xn--" in host:
            r.findings.append(URLFinding(
                code="PUNYCODE",
                message="Host uses punycode (IDN homograph attack vector).",
                weight=1.5,
            ))

    def _check_encoded_payload(self, parsed, r: URLAnalysisResult) -> None:
        raw = f"{parsed.path}?{parsed.query}"
        try:
            decoded = unquote(raw)
        except Exception:
            decoded = raw
        if "http://" in decoded or "https://" in decoded:
            r.findings.append(URLFinding(
                code="EMBEDDED_URL",
                message=(
                    "URL contains an embedded second URL "
                    "(possible open redirect)."
                ),
                weight=1.0,
            ))
        if raw.count("%") >= 6:
            r.findings.append(URLFinding(
                code="HEAVY_ENCODING",
                message=(
                    "Path/query is heavily percent-encoded — "
                    "obfuscation indicator."
                ),
                weight=0.6,
            ))

    def _check_telegram_impersonation(
        self, host: str, parsed, r: URLAnalysisResult,
    ) -> None:
        """
        Detect Telegram impersonation:
          Pattern 1 — legitimate t.me / telegram.* hosts with suspicious handles
                      (e.g. t.me/telebirr_official_support).
          Pattern 2 — lookalike hosts (telegram-verify.xyz, tme-login.top, …).
        """
        # -------- Pattern 1: real Telegram host, suspicious handle --------
        if host in {"t.me", "telegram.me", "telegram.org"}:
            try:
                handle = parsed.path.strip("/").split("/")[0].lower()
            except Exception:
                handle = ""
            if handle and _RE_TG_SUSPICIOUS_HANDLE.search(handle):
                r.findings.append(URLFinding(
                    code="TG_SUSPICIOUS_HANDLE",
                    message=(
                        f"Telegram handle '@{handle}' mimics an "
                        f"official/support channel."
                    ),
                    weight=1.5,
                ))
            return

        # -------- Pattern 2: lookalike Telegram host --------
        for tg_alias in ("telegram", "tme", "telegramme", "telegramapp"):
            if tg_alias not in host:
                continue
            # Confirm host is NOT the real Telegram
            if host.endswith("telegram.org") or host.endswith("t.me") \
                    or host.endswith("telegram.me"):
                continue
            r.impersonated_brand = r.impersonated_brand or "telegram"
            r.brand_sector = r.brand_sector or "telegram"
            r.findings.append(URLFinding(
                code="TELEGRAM_LOOKALIKE",
                message=f"Host '{host}' impersonates Telegram infrastructure.",
                weight=2.3,
            ))
            return

    def _check_entropy(self, host: str, r: URLAnalysisResult) -> None:
        """
        High Shannon entropy on the registered label suggests a
        machine-generated / DGA-style domain.
        """
        try:
            label = host.split(".")[0] if "." in host else host
            if len(label) < 8:
                return
            entropy = _shannon_entropy(label)
            if entropy >= self.entropy_threshold:
                r.findings.append(URLFinding(
                    code="HIGH_ENTROPY_HOST",
                    message=(
                        f"Host label '{label}' has high entropy "
                        f"({entropy:.2f} bits/char) — typical of "
                        f"machine-generated phishing domains."
                    ),
                    weight=1.2,
                ))
        except Exception:
            logger.exception("Entropy check failed for host %s", host)

    def _check_suspicious_patterns(self, url: str, r: URLAnalysisResult) -> None:
        """Apply pre-compiled regex patterns to the raw URL string."""
        for pattern, code, message, weight in SUSPICIOUS_PATTERNS:
            try:
                if pattern.search(url):
                    r.findings.append(URLFinding(
                        code=code, message=message, weight=weight,
                    ))
            except Exception:
                logger.exception("Regex '%s' failed on URL %s", code, url)

    # ------------------------------------------------------------------ #
    # SMALL UTILITIES
    # ------------------------------------------------------------------ #

    @staticmethod
    def _subdomain_depth(host: str, registered: str) -> int:
        if not host or not registered:
            return host.count(".")
        if host.endswith(registered):
            prefix = host[: -len(registered)].rstrip(".")
            return 0 if not prefix else prefix.count(".") + 1
        return host.count(".")