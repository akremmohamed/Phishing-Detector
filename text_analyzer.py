"""
text_analyzer.py
==================

Localized multi-context scam/phishing text detector with on-disk model
persistence via joblib.

On first start it trains TF-IDF (word + char n-grams) → LogisticRegression
on a curated trilingual corpus (Amharic / Afan Oromo / English) and dumps
the pipeline to `phishing_model.joblib`. On subsequent starts the cached
model is loaded instead of retraining.

Public API
----------
    analyzer = TextAnalyzer()
    result   = analyzer.analyze("እድለኛ ነዎት …")            # → TextAnalysisResult
    data     = analyzer.analyze_to_dict("…")              # → dict
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger("text_analyzer")


# ============================================================================
# MODEL PERSISTENCE CONFIG
# ============================================================================

MODEL_PATH = Path(os.getenv("PHISHING_MODEL_PATH", "phishing_model.joblib"))
MODEL_VERSION = 2  # bump to invalidate cached models after corpus changes


# ============================================================================
# LEXICONS  (Amharic / Afan Oromo / English)
# ============================================================================

AMHARIC_SCAM_KEYWORDS: Dict[str, float] = {
    "እድለኛ": 1.8, "እድለኛ ነዎት": 2.2, "ሎተሪ": 1.8,
    "ሎተሪ አሸንፈዋል": 2.5, "አሸንፈዋል": 1.8, "አሸንፈሃል": 1.8,
    "አሸንፋለህ": 1.8, "አሸንፏል": 1.5, "ሽልማት": 1.6,
    "ሽልማት አሸንፈዋል": 2.5, "የገንዘብ ሽልማት": 1.8,
    "ስጦታ": 1.0, "ጉርሻ": 0.8, "ተመርጠዋል": 1.4, "ተመርጠሃል": 1.4,
    "የይለፍ ቃል": 2.5, "የይለፍ ቃልዎን": 2.5, "የይለፍ ቃል ያስገቡ": 3.0,
    "ምስጢር ቁጥር": 2.5, "ምስጢር ቁጥርዎን": 2.5, "OTP": 2.5, "ኦቲፒ": 2.5,
    "ፒን": 1.8, "ፒን ኮድ": 2.5, "የመግቢያ ቁጥር": 1.8,
    "አካውንት ቁጥር": 1.5, "የባንክ አካውንት": 1.4,
    "አካውንትዎ ታግዷል": 3.0, "አካውንትዎ ተቋርጧል": 3.0,
    "አካውንትዎ ተዘግቷል": 3.0, "አካውንትዎ ይታገዳል": 2.8,
    "ታግዷል": 2.0, "ተቋርጧል": 2.0, "ተዘግቷል": 2.0,
    "አሁን ያረጋግጡ": 2.5, "ወዲያውኑ ያረጋግጡ": 2.8,
    "አስቸኳይ": 2.0, "አስቸኳይ ነው": 2.2, "ፈጣን": 1.0,
    "ገንዘብ ተልኳል": 1.8, "ገንዘብ ይላኩ": 2.0, "ብር ይላኩ": 2.0,
    "ያስተላልፉ": 1.8, "ተመላሽ ገንዘብ": 1.8, "ካሳ": 1.2,
    "የገንዘብ ድጋፍ": 1.5, "ልዩ ድጋፍ": 1.2, "ብር": 0.5,
    "ቴሌብር": 1.2, "ቴሌ ብር": 1.2, "ኢትዮ ቴሌኮም": 1.0,
    "ኢትዮጵያ ንግድ ባንክ": 1.2, "ዳሽን ባንክ": 1.2, "አዋሽ ባንክ": 1.2,
    "አቢሲኒያ ባንክ": 1.2, "ህብረት ባንክ": 1.2, "ዘመን ባንክ": 1.2,
    "ሊንክ ይጫኑ": 2.5, "ሊንኩን ይጫኑ": 2.8, "ይህን ሊንክ": 2.5,
    "ይህንን ሊንክ": 2.5, "ይጎብኙ": 1.6, "እዚህ ይጫኑ": 2.5,
    "በዚህ ሊንክ": 2.5, "ቴሌግራም": 1.0, "ቴለግራም": 1.0,
    "ዋትስ አፕ": 0.8,
    "24 ሰዓት": 1.0, "24 ሰዓታት": 1.0, "48 ሰዓት": 1.0,
    "ካለፈ": 1.5, "ካለቀ": 1.5,
    "ሕጋዊ እርምጃ": 2.5, "ይወሰድብዎታል": 2.5, "እርምጃ": 1.5,
    "ተወስዶብዎታል": 2.0, "ተጠያቂ": 1.5, "ኃላፊነት": 1.2,
    "ማንነትዎ": 1.8, "ማንነትዎን ያረጋግጡ": 2.5, "ማንነት ስርቆት": 1.8,
    "የሥራ ዕድል": 1.5, "የሥራ ዕድል አለ": 1.8, "ቅጥር": 1.5,
    "ቅጥር እናደርጋለን": 1.8, "በቤት ውስጥ ይሥሩ": 2.0,
    "የቤት ውስጥ ሥራ": 2.0, "ከቤት ይሥሩ": 2.0,
    "ደመወዝ": 1.5, "ደመወዝ 25000": 2.0, "በሳምንት": 1.2,
    "ሲቪ ይላኩ": 1.8, "CV ይላኩ": 1.8, "ማመልከቻ": 1.2,
    "የሥራ ቃለ መጠይቅ": 1.0, "ሠራተኛ እንፈልጋለን": 1.5,
    "ሠራተኞች እንፈልጋለን": 1.5, "ያለ ልምድ": 1.2,
    "ቪዛ": 1.8, "ቪዛ ዕድል": 2.0, "የሥራ ቪዛ": 2.0,
    "የትምህርት ዕድል": 1.8, "ስኮላርሺፕ": 1.8, "ስኮላርሺፕ ዕድል": 2.0,
    "የጥናት ዕድል": 1.8, "ነጻ ትምህርት": 1.5, "ወደ ውጭ ሀገር": 1.5,
    "ወደ ካናዳ": 1.5, "ወደ አሜሪካ": 1.5, "ወደ አውሮፓ": 1.5,
    "የጉዞ ወጪ": 1.5, "የቪዛ ክፍያ": 1.8, "ማመልከቻ ክፍያ": 1.5,
    "ኤጀንሲ": 1.0, "የጉዞ ኤጀንሲ": 1.2,
    "የቴሌግራም አካውንትዎ": 2.5, "የቴሌግራም አካውንትዎ ሊዘጋ ነው": 3.0,
    "የቴሌግራም አካውንትዎ ይዘጋል": 3.0,
    "የቴሌግራም አካውንትዎ ታግዷል": 3.0,
    "የቴሌግራም አካውንት": 2.0, "አካውንትዎ ሊዘጋ ነው": 2.5,
    "የቴሌግራም ማረጋገጫ": 2.2, "የቴሌግራም ኮድ": 2.2,
    "ቴሌግራም ኮድ ይላኩ": 2.8, "የማረጋገጫ ኮድ": 2.5,
    "የማረጋገጫ ኮድዎን ይላኩ": 3.0,
    "አካውንትዎን ለማረጋገጥ": 2.5,
    "ነጻ ገንዘብ": 2.2, "ነጻ ስጦታ": 1.8, "ልዩ ዕድል": 1.5,
    "የንግድ ዕድል": 1.2, "በቀላሉ ገንዘብ": 2.0,
    "ሀብታም ይሁኑ": 2.2, "ፈጣን ገንዘብ": 2.2,
}

AFAN_OROMO_SCAM_KEYWORDS: Dict[str, float] = {
    "carraa": 1.5, "carraan": 1.5, "carraa qabda": 2.2, "carraa argatte": 2.2,
    "lootarii": 2.0, "lootariin": 1.8, "mo'atte": 2.0, "mo'atani": 1.8,
    "kennaa": 1.2, "kennaa argatte": 2.0, "badhaasa": 1.5,
    "filatamte": 1.5, "filatamtan": 1.5, "abjuu": 1.0,
    "jecha iccitii": 2.5, "jecha darbii": 2.5, "jecha iccitii kee": 2.5,
    "lakkoofsa iccitii": 2.5, "OTP": 2.5, "lakkoofsa mirkaneessaa": 2.5,
    "PIN": 2.0, "koodii PIN": 2.5, "maqaa galmee": 1.5,
    "herrega baankii": 1.5, "lakkoofsa herregaa": 1.5,
    "herregni kee cufameera": 3.0, "herregni kee cufama": 2.8,
    "herregni kee dhaabbateera": 3.0, "herregni kee dhaabbata": 2.8,
    "cufameera": 2.0, "dhaabbateera": 2.0,
    "amuma mirkaneessi": 2.5, "saffisaan mirkaneessi": 2.8,
    "ariifachiisaa": 2.0, "ariifachiisaa dha": 2.2,
    "maallaqa ergameera": 1.8, "maallaqa ergi": 2.0, "maallaqa nuuf ergi": 2.2,
    "maallaqa deebisaa": 1.8, "bu'aa": 1.2, "deebii maallaqaa": 1.5,
    "telebirr": 1.2, "ethiotelecom": 1.0, "baankii daldalaa": 1.2,
    "baankii daashin": 1.2, "baankii awaash": 1.2, "baankii abisiiniyaa": 1.2,
    "baankii hibrat": 1.2, "baankii zamanaa": 1.2, "amole": 1.0,
    "liinkii tuqi": 2.5, "liinkii kana tuqi": 2.8, "as tuqi": 2.5,
    "liinkii armaan gadii": 2.5, "daawwadhu": 1.6,
    "telegram": 1.0, "telegraam": 1.0, "whatsapp": 0.8,
    "sa'aatii 24": 1.0, "guyyaa 2": 1.0, "yeroo gabaabaa": 1.5,
    "akkuma yaalame": 1.2,
    "tarkaanfii seeraa": 2.5, "tarkaanfii": 1.5,
    "itti gaafatama": 1.5, "eenyummaa kee": 1.8,
    "eenyummaa kee mirkaneessi": 2.5,
    "carraa hojii": 1.8, "hojii": 1.0, "hojii barbaadna": 1.5,
    "hojjetaa barbaadna": 1.5, "mana keessaa hojjedhu": 2.0,
    "mindaa": 1.5, "mindaa 25000": 2.0, "torbanitti": 1.2,
    "CV ergi": 1.8, "CV kee ergi": 1.8, "gaaffii hojii": 1.0,
    "muuxannoo malee": 1.2,
    "viizaa": 1.8, "carraa viizaa": 2.0, "viizaa hojii": 2.0,
    "carraa barnootaa": 1.8, "iskoolaarshiippii": 1.8,
    "barnoota bilisaa": 1.5, "biiraa alaa": 1.5,
    "Kaanaadaa": 1.5, "Ameerikaa": 1.5, "Awurooppaa": 1.5,
    "kaffaltii viizaa": 1.8, "kaffaltii iyyataa": 1.5, "ejensii": 1.0,
    "herrega telegram kee": 2.5, "herrega telegram kee cufama": 3.0,
    "herrega telegram kee cufameera": 3.0,
    "koodii telegram": 2.2, "koodii mirkaneessaa": 2.5,
    "koodii mirkaneessaa ergi": 3.0,
    "maallaqa bilisaa": 2.2, "kennaa bilisaa": 1.8,
    "saffisaan qabeenya": 2.2, "daddabalaan maallaqa": 2.0,
}

ENGLISH_SCAM_KEYWORDS: Dict[str, float] = {
    "congratulations": 1.5, "you have won": 2.5, "you've won": 2.5,
    "winner": 1.8, "lottery": 2.0, "lucky": 1.5, "prize": 1.8,
    "award": 1.5, "claim your": 2.2, "claim now": 2.5,
    "selected": 1.3, "chosen": 1.2,
    "password": 2.0, "otp": 2.5, "one-time password": 2.5,
    "one time password": 2.5, "pin": 2.0, "pin code": 2.5,
    "verify your account": 3.0, "verify account": 2.5,
    "confirm your": 2.0, "update your": 1.8,
    "login": 1.5, "sign in": 1.5, "credentials": 2.0,
    "urgent": 2.0, "immediately": 2.0, "within 24 hours": 2.5,
    "within 48 hours": 2.2, "account will be": 2.8,
    "account has been": 2.5, "suspended": 2.5, "blocked": 2.2,
    "deactivated": 2.5, "terminated": 2.5, "closed": 1.5,
    "legal action": 2.5, "act now": 2.5,
    "final warning": 2.8, "last warning": 2.8,
    "click here": 2.5, "click the link": 2.5, "click below": 2.5,
    "follow this link": 2.5, "visit this link": 2.5, "tap here": 2.5,
    "join telegram": 1.5, "whatsapp": 1.0, "bit.ly": 1.5, "tinyurl": 1.5,
    "free money": 2.5, "cash prize": 2.5, "transfer": 1.2,
    "refund": 1.5, "compensation": 1.5, "wire": 1.2,
    "western union": 1.8, "gift card": 2.0, "crypto": 1.5,
    "bitcoin": 1.5,
    "telebirr": 1.2, "ethiotelecom": 1.0, "ethio telecom": 1.0,
    "commercial bank of ethiopia": 1.2, "cbe": 1.0,
    "dashen bank": 1.2, "awash bank": 1.2,
    "bank of abyssinia": 1.2, "amole": 1.0, "m-pesa": 1.0,
    "we are hiring": 1.8, "now hiring": 1.8, "urgent hiring": 2.2,
    "work from home": 2.0, "remote job": 1.8, "earn from home": 2.2,
    "earn daily": 2.0, "earn weekly": 2.0, "weekly pay": 1.8,
    "no experience": 1.5, "no experience needed": 1.8,
    "send your cv": 1.8, "submit cv": 1.5, "apply now": 1.5,
    "salary": 1.2, "per week": 1.2, "part time": 1.0,
    "work online": 1.5, "flexible hours": 1.2,
    "jobs available": 1.5, "vacancy": 1.2,
    "visa": 1.8, "work visa": 2.0, "visa opportunity": 2.0,
    "scholarship": 1.8, "full scholarship": 2.0, "free scholarship": 2.0,
    "study abroad": 1.8, "free education": 1.5,
    "canada": 1.2, "usa": 1.2, "europe": 1.2, "australia": 1.2,
    "visa fee": 1.8, "application fee": 1.5, "processing fee": 1.8,
    "agency": 1.0, "travel agency": 1.2,
    "guaranteed visa": 2.5, "guaranteed admission": 2.5,
    "your telegram account": 2.5, "telegram account will be": 3.0,
    "telegram account has been": 3.0, "telegram account suspended": 3.0,
    "telegram verification code": 2.5, "telegram code": 2.2,
    "send verification code": 3.0, "send the code": 2.8,
    "verify telegram": 2.5, "telegram verification": 2.2,
    "free gift": 1.8, "special offer": 1.2, "exclusive offer": 1.5,
    "get rich": 2.2, "make money fast": 2.5, "easy money": 2.2,
    "limited time": 1.5, "act fast": 2.2, "hurry up": 1.5,
}

CATEGORY_KEYWORDS: Dict[str, Dict[str, float]] = {
    "FINANCIAL": {
        "ሎተሪ": 1.5, "አሸንፈዋል": 1.5, "ሽልማት": 1.3, "ብር": 0.4,
        "የይለፍ ቃል": 1.5, "OTP": 1.5, "ፒን": 1.2, "ቴሌብር": 1.0,
        "አካውንት": 1.0,
        "lootarii": 1.5, "mo'atte": 1.5, "maallaqa": 0.8,
        "jecha iccitii": 1.5,
        "lottery": 1.5, "prize": 1.3, "won": 1.2, "cash": 1.0,
        "otp": 1.5, "pin": 1.2, "wallet": 1.0, "transfer": 1.0,
    },
    "JOB_SCAM": {
        "የሥራ ዕድል": 1.5, "ቅጥር": 1.3, "ደመወዝ": 1.3,
        "በቤት ውስጥ ይሥሩ": 1.8, "ከቤት ይሥሩ": 1.8,
        "ሲቪ ይላኩ": 1.5, "ሠራተኛ እንፈልጋለን": 1.3,
        "carraa hojii": 1.8, "hojii barbaadna": 1.5,
        "mana keessaa hojjedhu": 1.8, "CV ergi": 1.5, "mindaa": 1.3,
        "hiring": 1.5, "work from home": 1.8, "remote job": 1.5,
        "weekly pay": 1.5, "no experience": 1.5, "send your cv": 1.5,
        "apply now": 1.2, "salary": 1.0, "vacancy": 1.2,
    },
    "VISA_SCHOLARSHIP": {
        "ቪዛ": 1.5, "ስኮላርሺፕ": 1.5, "የትምህርት ዕድል": 1.5,
        "የጉዞ ወጪ": 1.3, "የቪዛ ክፍያ": 1.5, "ኤጀንሲ": 1.0,
        "viizaa": 1.5, "iskoolaarshiippii": 1.5, "barnoota bilisaa": 1.3,
        "kaffaltii viizaa": 1.5, "ejensii": 1.0,
        "visa": 1.5, "scholarship": 1.5, "study abroad": 1.5,
        "visa fee": 1.5, "application fee": 1.3, "agency": 1.0,
        "guaranteed visa": 2.0,
    },
    "TELEGRAM_ACCOUNT_TAKEOVER": {
        "የቴሌግራም አካውንት": 1.8, "የቴሌግራም አካውንትዎ ሊዘጋ ነው": 2.2,
        "የማረጋገጫ ኮድ": 1.8, "ቴሌግራም ኮድ": 1.8,
        "herrega telegram": 1.8, "koodii telegram": 1.8,
        "koodii mirkaneessaa": 1.8,
        "telegram account": 1.8, "telegram code": 1.8,
        "verification code": 1.5, "verify telegram": 1.8,
        "send the code": 1.8,
    },
    "CREDENTIAL_HARVEST": {
        "የይለፍ ቃል": 2.0, "ምስጢር ቁጥር": 2.0, "ፒን": 1.5,
        "ኦቲፒ": 2.0, "የማረጋገጫ ኮድ": 1.8,
        "jecha iccitii": 2.0, "jecha darbii": 2.0,
        "lakkoofsa iccitii": 2.0, "koodii PIN": 1.5,
        "password": 1.8, "otp": 2.0, "pin code": 1.8,
        "verification code": 1.5, "credentials": 1.5, "login": 1.2,
    },
}

# Pre-compiled structural patterns
STRUCTURAL_PATTERNS: List[Tuple[re.Pattern, str, str, float, str]] = [
    (re.compile(r"https?://\S+"), "CONTAINS_URL",
     "Message contains a URL.", 0.6, "GENERAL_SCAM"),
    (re.compile(r"t\.me/\S+"), "TELEGRAM_LINK",
     "Contains a Telegram t.me link.", 1.2, "TELEGRAM_ACCOUNT_TAKEOVER"),
    (re.compile(r"wa\.me/\S+"), "WHATSAPP_LINK",
     "Contains a WhatsApp wa.me link.", 1.0, "GENERAL_SCAM"),
    (re.compile(r"bit\.ly|tinyurl|goo\.gl|ow\.ly|shorturl|rb\.gy|cutt\.ly"),
     "SHORTENER", "Message uses a URL shortener.", 1.2, "GENERAL_SCAM"),
    (re.compile(r"\b\d{9,16}\b"), "LONG_NUMBER",
     "Contains a long numeric sequence (possibly account/phone).", 0.5, "FINANCIAL"),
    (re.compile(r"\+251\d{9}"), "PHONE_ET",
     "Contains an Ethiopian phone number.", 0.3, "GENERAL_SCAM"),
    (re.compile(r"[0-9]{3,8}\s*(ብር|birr|ETB)"), "MONEY_AMOUNT",
     "Contains an explicit money amount.", 0.5, "FINANCIAL"),
    (re.compile(r"\b(OTP|otp)\b.{0,30}\d"), "OTP_WITH_DIGITS",
     "Contains OTP alongside digits.", 1.5, "CREDENTIAL_HARVEST"),
    (re.compile(r"@[a-zA-Z0-9_]{4,32}"), "TELEGRAM_HANDLE",
     "Contains a Telegram-style @handle.", 0.6, "TELEGRAM_ACCOUNT_TAKEOVER"),
    (re.compile(r"\b\d+\s*(USD|EUR|GBP|CAD|AUD)\b"), "FOREIGN_MONEY",
     "Contains a foreign currency amount.", 0.5, "VISA_SCHOLARSHIP"),
    (re.compile(r"(visa)\b.*\b(fee|payment|pay|processing)", re.IGNORECASE),
     "VISA_FEE_LURE", "Mentions visa fee/payment.", 1.5, "VISA_SCHOLARSHIP"),
    (re.compile(
        r"\b(whatsapp|telegram)\b.*\b(join|contact|message|dm|inbox)\b",
        re.IGNORECASE),
     "SOCIAL_CTA", "Directs user to contact via social app.", 1.0, "GENERAL_SCAM"),
]

# Regexes for Amharic-script detection (compiled once)
_RE_AMHARIC_CHAR = re.compile(r"[\u1200-\u137F]")
_RE_LATIN_CHAR = re.compile(r"[a-zA-Z]")

# Training corpus — (text, is_scam, category_or_None)
TRAINING_CORPUS: List[Tuple[str, int, Optional[str]]] = [
    # ---- Amharic — FINANCIAL ----
    ("እንኳን ደስ አለዎት! በቴሌብር ሎተሪ 50,000 ብር አሸንፈዋል። ሊንክ ይጫኑ።", 1, "FINANCIAL"),
    ("እድለኛ ነዎት! የቴሌብር ሽልማት አሸንፈዋል። ወዲያውኑ ያረጋግጡ።", 1, "FINANCIAL"),
    ("የይለፍ ቃልዎን ያስገቡ አለበለዚያ አካውንትዎ ታግዷል።", 1, "FINANCIAL"),
    ("አካውንትዎ ተቋርጧል። ምስጢር ቁጥርዎን ይላኩልን።", 1, "FINANCIAL"),
    ("የCBE አካውንትዎ ታግዷል። ይህን ሊንክ ይጫኑ እና ያረጋግጡ።", 1, "FINANCIAL"),
    ("OTP ኮድዎን ለዚህ ቴሌግራም ቻናል ይላኩ።", 1, "CREDENTIAL_HARVEST"),
    ("በዳሽን ባንክ ስም 10,000 ብር ስጦታ አሸንፈዋል። ሊንክ ይጫኑ።", 1, "FINANCIAL"),
    ("አስቸኳይ! የአዋሽ ባንክ አካውንትዎ ይዘጋል። አሁን ያረጋግጡ።", 1, "FINANCIAL"),
    ("እድለኛ ነዎት! ሽልማትዎን ለመቀበል የባንክ አካውንትዎን ያረጋግጡ።", 1, "FINANCIAL"),
    ("የቴሌብር ሚስጥራዊ ቁጥርዎን ለሌላ ሰው አይስጡ - ይላኩልን።", 1, "CREDENTIAL_HARVEST"),
    ("የማንነትዎን ማረጋገጫ ያስፈልጋል። ፒን ኮድ ያስገቡ።", 1, "CREDENTIAL_HARVEST"),
    ("ሕጋዊ እርምጃ ይወሰድብዎታል ካልተከፈለ። ወዲያውኑ ይክፈሉ።", 1, "FINANCIAL"),
    ("በ24 ሰዓት ውስጥ ያረጋግጡ አለበለዚያ አካውንትዎ ተዘግቷል።", 1, "FINANCIAL"),
    ("አሸንፈዋል! ሽልማትዎን ለማግኘት ይህን ሊንክ ይጫኑ።", 1, "FINANCIAL"),
    ("የባንክ ካርድዎ ተቋርጧል። ወዲያውኑ ያረጋግጡ።", 1, "FINANCIAL"),
    ("እንኳን ደስ አለዎት በህብረት ባንክ የገንዘብ ሽልማት አሸንፈዋል።", 1, "FINANCIAL"),
    ("የዘመን ባንክ ደንበኛ ነዎት? የይለፍ ቃልዎን ያስገቡ።", 1, "CREDENTIAL_HARVEST"),
    ("ሎተሪ አሸንፈዋል! ሊንኩን ይጫኑ እና መረጃዎን ያስገቡ።", 1, "FINANCIAL"),
    ("የአቢሲኒያ ባንክ ስም ስለተጠቀመ ማንነትዎን ያረጋግጡ።", 1, "CREDENTIAL_HARVEST"),
    ("አስቸኳይ መልእክት! OTP ኮድዎን ወዲያውኑ ይላኩ።", 1, "CREDENTIAL_HARVEST"),

    # ---- Amharic — JOB SCAM ----
    ("የሥራ ዕድል! በቤት ውስጥ ይሥሩ፣ በሳምንት 25,000 ብር ያግኙ። CV ይላኩ።", 1, "JOB_SCAM"),
    ("ሠራተኛ እንፈልጋለን! ያለ ልምድ ቅጥር። ሲቪ ወደ ቴሌግራም ይላኩ።", 1, "JOB_SCAM"),
    ("አስቸኳይ ቅጥር! ደመወዝ 30,000 ብር። ዝርዝር ለማግኘት ሊንክ ይጫኑ።", 1, "JOB_SCAM"),
    ("የቤት ውስጥ ሥራ ዕድል - በቀላሉ ገንዘብ ያግኙ። ዛሬ ይመዝገቡ።", 1, "JOB_SCAM"),

    # ---- Amharic — VISA / SCHOLARSHIP ----
    ("የቪዛ ዕድል! ወደ ካናዳ የሥራ ቪዛ። የቪዛ ክፍያ ብቻ ይክፈሉ።", 1, "VISA_SCHOLARSHIP"),
    ("ነጻ ስኮላርሺፕ ዕድል! ወደ አውሮፓ የጥናት ዕድል። ማመልከቻ ክፍያ ያስፈልጋል።", 1, "VISA_SCHOLARSHIP"),
    ("የትምህርት ዕድል በአሜሪካ! ኤጀንሲያችን ዋስትና ይሰጣል።", 1, "VISA_SCHOLARSHIP"),

    # ---- Amharic — TELEGRAM TAKEOVER ----
    ("የቴሌግራም አካውንትዎ ሊዘጋ ነው! የማረጋገጫ ኮድ ይላኩ።", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),
    ("የቴሌግራም አካውንትዎ ታግዷል። አሁን ለማረጋገጥ ኮድ ይላኩ።", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),
    ("የቴሌግራም ማረጋገጫ ያስፈልጋል። የማረጋገጫ ኮድዎን ይላኩልን።", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),

    # ---- Afan Oromo — FINANCIAL ----
    ("Baga gammadde! Lootarii telebirr keessatti qarshii 50,000 mo'atte. Liinkii tuqi.", 1, "FINANCIAL"),
    ("Carraa qabda! Kennaa telebirr argatte. Amuma mirkaneessi.", 1, "FINANCIAL"),
    ("Herregni baankii kee cufameera. Jecha iccitii kee nuuf ergi.", 1, "FINANCIAL"),
    ("Herregni kee dhaabbateera. Lakkoofsa iccitii kee ergi.", 1, "FINANCIAL"),
    ("OTP koodii kee kanaaf telegram ergi.", 1, "CREDENTIAL_HARVEST"),
    ("Ariifachiisaa! Herregni baankii awaash kee ni cufama. Amuma mirkaneessi.", 1, "FINANCIAL"),
    ("Maallaqa ergameera. Herrega kee mirkaneessi.", 1, "FINANCIAL"),

    # ---- Afan Oromo — JOB ----
    ("Carraa hojii! Mana keessaa hojjedhu, torbanitti qarshii 25,000 argadhu. CV ergi.", 1, "JOB_SCAM"),
    ("Hojjetaa barbaadna! Muuxannoo malee. CV kee telegram ergi.", 1, "JOB_SCAM"),
    ("Hojii saffisaa! Mindaa 30,000. Liinkii tuqi.", 1, "JOB_SCAM"),

    # ---- Afan Oromo — VISA ----
    ("Carraa viizaa! Viizaa hojii Kaanaadaa. Kaffaltii viizaa qofa kaffali.", 1, "VISA_SCHOLARSHIP"),
    ("Iskoolaarshiippii bilisaa! Awurooppaa barnoota. Kaffaltii iyyataa barbaachisa.", 1, "VISA_SCHOLARSHIP"),

    # ---- Afan Oromo — TELEGRAM ----
    ("Herrega telegram kee cufama! Koodii mirkaneessaa ergi.", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),
    ("Herrega telegram kee cufameera. Amuma mirkaneessi.", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),

    # ---- English — FINANCIAL ----
    ("Congratulations! You have won 50,000 ETB in the telebirr lottery. Click here to claim.", 1, "FINANCIAL"),
    ("URGENT: Your CBE account has been suspended. Verify your account immediately.", 1, "FINANCIAL"),
    ("You are the lucky winner! Send your OTP to claim your prize now.", 1, "CREDENTIAL_HARVEST"),
    ("Your Dashen Bank account will be closed within 24 hours. Click here to verify.", 1, "FINANCIAL"),
    ("Final warning: Your account has been blocked. Confirm your PIN code now.", 1, "FINANCIAL"),
    ("You have won a cash prize of 100,000 birr. Click the link to claim your award.", 1, "FINANCIAL"),
    ("Verify your account now or legal action will be taken. Act immediately.", 1, "FINANCIAL"),
    ("Enter your password and OTP at http://telebirr-verify.xyz to unlock your account.", 1, "CREDENTIAL_HARVEST"),
    ("Your Awash Bank account is suspended. Update your credentials immediately.", 1, "FINANCIAL"),
    ("Congratulations winner! Selected for Amole wallet reward. Claim now via bit.ly link.", 1, "FINANCIAL"),
    ("Your Bank of Abyssinia account is deactivated. Confirm your details here.", 1, "FINANCIAL"),
    ("Last warning: Your Ethio Telecom line will be terminated. Verify your identity.", 1, "FINANCIAL"),
    ("Click below to receive your refund from Commercial Bank of Ethiopia.", 1, "FINANCIAL"),
    ("You are chosen for a free money transfer. Send your account number and PIN.", 1, "CREDENTIAL_HARVEST"),
    ("Urgent: Verify your telebirr account within 48 hours to avoid suspension.", 1, "FINANCIAL"),
    ("Your Hibret Bank account is blocked. Visit this link to reactivate.", 1, "FINANCIAL"),
    ("Claim your 5000 birr gift card from Zemen Bank. Click here now.", 1, "FINANCIAL"),

    # ---- English — JOB ----
    ("We are hiring! Work from home and earn 25,000 birr per week. Send your CV to t.me/jobs_et", 1, "JOB_SCAM"),
    ("Urgent hiring! No experience needed. Weekly pay. Apply now.", 1, "JOB_SCAM"),
    ("Remote job available. Earn daily from home. Send CV via WhatsApp.", 1, "JOB_SCAM"),
    ("Part time work online. Flexible hours. Contact us on Telegram.", 1, "JOB_SCAM"),
    ("Vacancy: Salary 30,000 birr. Submit CV now. Limited time.", 1, "JOB_SCAM"),

    # ---- English — VISA ----
    ("Visa opportunity to Canada! Work visa. Pay only the visa fee.", 1, "VISA_SCHOLARSHIP"),
    ("Full scholarship to USA! Free education. Application fee required.", 1, "VISA_SCHOLARSHIP"),
    ("Guaranteed visa to Europe. Travel agency processing. Limited slots.", 1, "VISA_SCHOLARSHIP"),
    ("Study abroad in Australia! Scholarship available. Processing fee applies.", 1, "VISA_SCHOLARSHIP"),

    # ---- English — TELEGRAM ----
    ("Your Telegram account will be closed! Send verification code now.", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),
    ("Your Telegram account has been suspended. Verify Telegram by sending the code.", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),
    ("Telegram verification required. Send the code to us immediately.", 1, "TELEGRAM_ACCOUNT_TAKEOVER"),

    # ---- Amharic — HAM ----
    ("ሰላም ወዳጄ፤ ዛሬ ምሳ እንገናኝ?", 0, None),
    ("የቴሌብር ሂሳብዎ በተሳካ ሁኔታ ተከፍሏል። አመሰግናለሁ።", 0, None),
    ("የእናትህ ስልክ ቁጥር ምንድነው?", 0, None),
    ("ነገ በ9 ሰዓት ስብሰባ አለን።", 0, None),
    ("የቤት ስራውን ጨርሰሃል?", 0, None),
    ("የባንክ ሂሳብ ማጠቃለያ ወርሃዊ ሪፖርትዎ ዝግጁ ነው።", 0, None),
    ("እንዴት ነህ? ቤተሰብህ እንዴት ነው?", 0, None),
    ("ዛሬ የአየር ሁኔታ ጥሩ ነው።", 0, None),
    ("የስራ ቃለ መጠይቅ ቀን ተይዟል።", 0, None),
    ("አመሰግናለሁ ለእርዳታህ።", 0, None),
    ("የመኪና ነዳጅ ዋጋ ጨምሯል።", 0, None),
    ("ልጆቹ ትምህርት ቤት ሄዱ።", 0, None),

    # ---- Afan Oromo — HAM ----
    ("Akkam, boru laaqana wal argaa?", 0, None),
    ("Kaffaltiin telebirr kee milkaa'eera. Galatoomi.", 0, None),
    ("Lakkoofsa bilbilaa haadha kee maali?", 0, None),
    ("Boru sa'aatii 9 walga'ii qabna.", 0, None),
    ("Hojii mana kee xumurteettaa?", 0, None),
    ("Gabaasni ji'aa herrega baankii kee qophiidha.", 0, None),
    ("Akkam jirta? Maatiin kee akkam jiru?", 0, None),
    ("Haala qilleensa har'a gaarii dha.", 0, None),
    ("Guyyaan gaaffii hojii qabameera.", 0, None),
    ("Gargaarsa keetiif galatoomi.", 0, None),
    ("Gatii bobaa konkolaataa dabaleera.", 0, None),
    ("Ijoolleen mana barumsaa dhaqan.", 0, None),

    # ---- English — HAM ----
    ("Hi, are we still meeting for lunch tomorrow?", 0, None),
    ("Your telebirr payment of 500 ETB was successful. Thank you.", 0, None),
    ("Please send me your address when you get a chance.", 0, None),
    ("The monthly bank statement is now available in your app.", 0, None),
    ("Team meeting scheduled for 9 AM tomorrow.", 0, None),
    ("How was your weekend? Mine was relaxing.", 0, None),
    ("Your order has been shipped and will arrive on Tuesday.", 0, None),
    ("Can you review the document I sent earlier?", 0, None),
    ("The weather forecast says rain this afternoon.", 0, None),
    ("Thanks for your help with the project yesterday.", 0, None),
    ("Your electricity bill for this month is 350 birr.", 0, None),
    ("Reminder: doctor's appointment on Friday at 3 PM.", 0, None),
    ("We are hiring a new accountant at our office. Please share the JD.", 0, None),
    ("I applied for a scholarship at Addis Ababa University.", 0, None),
    ("Visa application submitted successfully. Waiting for confirmation.", 0, None),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TextFinding:
    code: str
    message: str
    weight: float
    category_hint: str = "GENERAL_SCAM"


@dataclass
class TextAnalysisResult:
    is_phishing: bool
    confidence: float
    risk_level: str
    scam_category: str
    category_scores: List[Tuple[str, float]] = field(default_factory=list)
    findings: List[TextFinding] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    language_detected: str = "unknown"
    ml_probability: float = 0.0
    lexicon_score: float = 0.0
    structural_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_phishing": self.is_phishing,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "scam_category": self.scam_category,
            "category_scores": [
                {"category": c, "score": s} for c, s in self.category_scores
            ],
            "reasons": [
                {"code": f.code, "message": f.message, "weight": round(f.weight, 3)}
                for f in self.findings
            ],
            "matched_keywords": self.matched_keywords,
            "language_detected": self.language_detected,
            "ml_probability": self.ml_probability,
            "lexicon_score": self.lexicon_score,
            "structural_score": self.structural_score,
        }


# ============================================================================
# SCRIPT / LANGUAGE HELPERS
# ============================================================================

def _amharic_ratio(text: str) -> float:
    if not text:
        return 0.0
    amharic = len(_RE_AMHARIC_CHAR.findall(text))
    total_alpha = amharic + len(_RE_LATIN_CHAR.findall(text))
    return amharic / max(total_alpha, 1)


def _oromo_latin_markers(text: str) -> int:
    """Afan Oromo-specific digraphs / suffixes to distinguish from English."""
    lower = text.lower()
    markers = (
        " kee", " keessatti", " jedhu", " irraa", " dhaan", " dhaaf",
        " qabda", " argatte", " mo'atte", " hojii", " maallaqa",
        " herrega", " koodii", " mirkaneessi", " ergi", " tuqi",
        " carraa", " kennaa", " barnoota", " viizaa", " mindaa",
        " akkam", " galatoomi", " jirta", " jira", " dha ", " kana ",
        "waan", "kanaaf", "itti", "kan", "hin", "malee",
    )
    return sum(1 for m in markers if m in lower)


# ============================================================================
# ANALYZER
# ============================================================================

class TextAnalyzer:
    """
    Multi-context scam text detector with on-disk model persistence.

    On construction:
      1. If `phishing_model.joblib` exists with a matching MODEL_VERSION,
         load it via joblib.
      2. Otherwise, train the pipeline on TRAINING_CORPUS and persist it.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        force_retrain: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.pipeline: Pipeline = self._load_or_train(force_retrain=force_retrain)

    # ------------------------------------------------------------------ #
    # MODEL PERSISTENCE
    # ------------------------------------------------------------------ #

    def _load_or_train(self, force_retrain: bool = False) -> Pipeline:
        """Load cached model if valid, otherwise train + save."""
        if not force_retrain and self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                if (
                    isinstance(payload, dict)
                    and payload.get("version") == MODEL_VERSION
                    and isinstance(payload.get("pipeline"), Pipeline)
                ):
                    logger.info(
                        "Loaded cached model from %s (trained_at=%s)",
                        self.model_path,
                        payload.get("trained_at", "unknown"),
                    )
                    return payload["pipeline"]
                logger.warning(
                    "Cached model at %s is incompatible (version mismatch); retraining.",
                    self.model_path,
                )
            except Exception:
                logger.exception(
                    "Failed to load cached model at %s; retraining.",
                    self.model_path,
                )

        logger.info("Training fresh model …")
        pipeline = self._train_pipeline()
        self._save_pipeline(pipeline)
        return pipeline

    def _train_pipeline(self) -> Pipeline:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
        )
        union = FeatureUnion([("word", word_vec), ("char", char_vec)])
        clf = LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )
        pipeline = Pipeline([("features", union), ("clf", clf)])

        texts = [t for t, _, _ in TRAINING_CORPUS]
        labels = np.array([y for _, y, _ in TRAINING_CORPUS])
        pipeline.fit(texts, labels)
        return pipeline

    def _save_pipeline(self, pipeline: Pipeline) -> None:
        """Persist the trained pipeline. Never raises."""
        try:
            payload = {
                "version": MODEL_VERSION,
                "pipeline": pipeline,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, self.model_path)
            logger.info("Saved model to %s", self.model_path)
        except Exception:
            logger.exception(
                "Could not save model to %s — continuing with in-memory model.",
                self.model_path,
            )

    def retrain(self, force: bool = False) -> None:
        """Public hook to force a retrain and re-save the pipeline."""
        if force or not self.model_path.exists():
            self.pipeline = self._train_pipeline()
            self._save_pipeline(self.pipeline)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> TextAnalysisResult:
        text = text.strip()
        amharic_ratio = _amharic_ratio(text)
        oromo_markers = _oromo_latin_markers(text)
        language = self._detect_language(text, amharic_ratio, oromo_markers)

        # ML inference (defensive — never crash the request)
        try:
            ml_prob = float(self.pipeline.predict_proba([text])[0][1])
        except Exception:
            logger.exception("ML inference failed; falling back to lexicon-only.")
            ml_prob = 0.0

        lexicon_findings, matched = self._lexicon_scan(text)
        structural_findings = self._structural_scan(text)

        lexicon_score = sum(f.weight for f in lexicon_findings)
        structural_score = sum(f.weight for f in structural_findings)

        lexicon_norm = min(1.0, lexicon_score / 7.0)
        structural_norm = min(1.0, structural_score / 3.0)

        combined = 0.55 * ml_prob + 0.30 * lexicon_norm + 0.15 * structural_norm

        if ml_prob > 0.6 and lexicon_norm > 0.5:
            combined = min(1.0, combined + 0.08)
        if ml_prob > 0.75 and lexicon_norm > 0.7:
            combined = min(1.0, combined + 0.07)

        combined = round(combined, 4)
        is_phishing = combined >= 0.5
        risk_level = self._risk_level(combined)

        category, category_scores = self._classify_category(
            text, lexicon_findings, structural_findings,
        )

        findings = lexicon_findings + structural_findings
        if ml_prob >= 0.65:
            findings.insert(0, TextFinding(
                code="ML_HIGH_RISK",
                message=f"ML classifier flagged this message as scam-like (p={ml_prob:.2f}).",
                weight=ml_prob,
                category_hint=category,
            ))
        elif ml_prob <= 0.25 and not findings:
            findings.append(TextFinding(
                code="ML_LOW_RISK",
                message=f"ML classifier considers this message benign (p={ml_prob:.2f}).",
                weight=0.0,
                category_hint="BENIGN",
            ))

        return TextAnalysisResult(
            is_phishing=is_phishing,
            confidence=combined,
            risk_level=risk_level,
            scam_category=category,
            category_scores=category_scores,
            findings=findings,
            matched_keywords=sorted(set(matched)),
            language_detected=language,
            ml_probability=round(ml_prob, 4),
            lexicon_score=round(lexicon_score, 3),
            structural_score=round(structural_score, 3),
        )

    def analyze_to_dict(self, text: str) -> Dict[str, Any]:
        return self.analyze(text).to_dict()

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_language(
        text: str, amharic_ratio: float, oromo_markers: int,
    ) -> str:
        has_latin = bool(_RE_LATIN_CHAR.search(text))
        if amharic_ratio > 0.35:
            return "amharic"
        if amharic_ratio > 0.08 and has_latin:
            return "amharic_english_mixed"
        if has_latin and oromo_markers >= 2:
            return "afan_oromo"
        if has_latin:
            return "english"
        return "unknown"

    def _lexicon_scan(
        self, text: str,
    ) -> Tuple[List[TextFinding], List[str]]:
        findings: List[TextFinding] = []
        matched: List[str] = []

        amharic_hits, oromo_hits, english_hits = self._keyword_hits(text)

        matched.extend(kw for kw, _ in amharic_hits)
        matched.extend(kw for kw, _ in oromo_hits)
        matched.extend(kw for kw, _ in english_hits)

        if amharic_hits:
            findings.append(TextFinding(
                code="AMHARIC_SCAM_LEXICON",
                message=(
                    f"Matched {len(amharic_hits)} Amharic scam keyword(s): "
                    f"{', '.join(k for k, _ in amharic_hits[:6])}"
                ),
                weight=sum(w for _, w in amharic_hits),
                category_hint=self._dominant_category_from_lexicon(amharic_hits),
            ))
        if oromo_hits:
            findings.append(TextFinding(
                code="AFAN_OROMO_SCAM_LEXICON",
                message=(
                    f"Matched {len(oromo_hits)} Afan Oromo scam keyword(s): "
                    f"{', '.join(k for k, _ in oromo_hits[:6])}"
                ),
                weight=sum(w for _, w in oromo_hits),
                category_hint=self._dominant_category_from_lexicon(oromo_hits),
            ))
        if english_hits:
            findings.append(TextFinding(
                code="ENGLISH_SCAM_LEXICON",
                message=(
                    f"Matched {len(english_hits)} English scam keyword(s): "
                    f"{', '.join(k for k, _ in english_hits[:6])}"
                ),
                weight=sum(w for _, w in english_hits),
                category_hint=self._dominant_category_from_lexicon(english_hits),
            ))
        return findings, matched

    @staticmethod
    def _keyword_hits(text: str) -> Tuple[
        List[Tuple[str, float]],
        List[Tuple[str, float]],
        List[Tuple[str, float]],
    ]:
        amharic_hits: List[Tuple[str, float]] = [
            (kw, w) for kw, w in AMHARIC_SCAM_KEYWORDS.items() if kw in text
        ]

        lower = text.lower()
        oromo_hits: List[Tuple[str, float]] = [
            (kw, w) for kw, w in AFAN_OROMO_SCAM_KEYWORDS.items()
            if kw.lower() in lower
        ]

        english_hits: List[Tuple[str, float]] = []
        for kw, w in ENGLISH_SCAM_KEYWORDS.items():
            k = kw.lower()
            if len(k) <= 4:
                # word-boundary match for short tokens
                if re.search(rf"\b{re.escape(k)}\b", lower):
                    english_hits.append((kw, w))
            else:
                if k in lower:
                    english_hits.append((kw, w))

        return amharic_hits, oromo_hits, english_hits

    @staticmethod
    def _dominant_category_from_lexicon(
        hits: List[Tuple[str, float]],
    ) -> str:
        keyword_set = {k for k, _ in hits}
        scores: Dict[str, float] = {}
        for cat, kw_map in CATEGORY_KEYWORDS.items():
            scores[cat] = sum(w for kw, w in kw_map.items() if kw in keyword_set)
        if not scores:
            return "GENERAL_SCAM"
        cat, sc = max(scores.items(), key=lambda kv: kv[1])
        return cat if sc > 0 else "GENERAL_SCAM"

    @staticmethod
    def _structural_scan(text: str) -> List[TextFinding]:
        findings: List[TextFinding] = []
        for pattern, code, message, weight, hint in STRUCTURAL_PATTERNS:
            try:
                if pattern.search(text):
                    findings.append(TextFinding(
                        code=code, message=message, weight=weight,
                        category_hint=hint,
                    ))
            except Exception:
                logger.exception("Structural regex '%s' failed.", code)
        return findings

    def _classify_category(
        self,
        text: str,
        lexicon_findings: List[TextFinding],
        structural_findings: List[TextFinding],
    ) -> Tuple[str, List[Tuple[str, float]]]:
        scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_KEYWORDS}
        scores["GENERAL_SCAM"] = 0.0
        scores["BENIGN"] = 0.0

        lower = text.lower()

        # Category keyword accumulation
        for cat, kw_map in CATEGORY_KEYWORDS.items():
            for kw, w in kw_map.items():
                if _RE_AMHARIC_CHAR.search(kw):
                    if kw in text:
                        scores[cat] += w
                else:
                    if kw.lower() in lower:
                        scores[cat] += w

        # Structural / lexicon hints
        for f in structural_findings:
            if f.category_hint in scores:
                scores[f.category_hint] += f.weight * 0.5
        for f in lexicon_findings:
            if f.category_hint in scores:
                scores[f.category_hint] += f.weight * 0.15

        # Cross-signal boosts
        if re.search(r"(t\.me/|telegram|ቴሌግራም|ቴለግራም|telegraam)", lower):
            if re.search(
                r"(ኮድ|code|koodii|verification|ማረጋገጫ|mirkaneessaa)", lower,
            ):
                scores["TELEGRAM_ACCOUNT_TAKEOVER"] += 2.0

        job_signals = sum(
            1 for kw in (
                "hiring", "job", "salary", "cv", "work from home",
                "ሥራ", "ቅጥር", "ደመወዝ", "ሲቪ",
                "hojii", "mindaa", "carraa hojii",
            ) if kw in lower
        )
        if job_signals >= 2:
            scores["JOB_SCAM"] += 1.0

        visa_signals = sum(
            1 for kw in (
                "visa", "scholarship", "study abroad",
                "ቪዛ", "ስኮላርሺፕ", "viizaa", "iskoolaarshiippii",
            ) if kw in lower
        )
        if visa_signals >= 1:
            scores["VISA_SCHOLARSHIP"] += 0.5
        if visa_signals >= 2:
            scores["VISA_SCHOLARSHIP"] += 0.5

        cred_signals = sum(
            1 for kw in (
                "otp", "pin", "password", "verification code",
                "የይለፍ ቃል", "ምስጢር ቁጥር", "ኦቲፒ",
                "jecha iccitii", "lakkoofsa iccitii", "koodii pin",
            ) if kw in lower or kw in text
        )
        if cred_signals >= 1:
            scores["CREDENTIAL_HARVEST"] += 0.8
        if cred_signals >= 2:
            scores["CREDENTIAL_HARVEST"] += 0.8

        fin_signals = sum(
            1 for kw in (
                "telebirr", "cbe", "dashen", "awash", "bank", "wallet",
                "lottery", "prize", "birr", "ብር", "ሎተሪ", "ሽልማት",
                "lootarii", "maallaqa", "baankii",
            ) if kw in lower or kw in text
        )
        if fin_signals >= 2:
            scores["FINANCIAL"] += 0.8

        # Fallback
        total = sum(v for k, v in scores.items() if k != "BENIGN")
        if total <= 0.0:
            scores["GENERAL_SCAM"] = 0.5

        ranked = sorted(
            [(cat, round(v, 3)) for cat, v in scores.items() if v > 0],
            key=lambda kv: kv[1], reverse=True,
        )
        if not ranked:
            return "GENERAL_SCAM", [("GENERAL_SCAM", 0.0)]

        top_cat, top_score = ranked[0]
        if top_score < 0.5:
            top_cat = "GENERAL_SCAM"
        return top_cat, ranked[:5]

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob >= 0.85:
            return "CRITICAL"
        if prob >= 0.70:
            return "HIGH"
        if prob >= 0.45:
            return "MEDIUM"
        return "LOW"