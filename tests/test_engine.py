"""
Smoke tests for the generalized phishing detection engine.
Run: pytest -q tests/
"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert "afan_oromo" in body["supported_languages"]
    assert "TELEGRAM_ACCOUNT_TAKEOVER" in body["supported_categories"]


# ------------------------------- TEXT ---------------------------------------

def test_text_amharic_lottery():
    r = client.post("/api/v1/scan-text", json={
        "text": "እድለኛ ነዎት! በቴሌብር ሎተሪ 50,000 ብር አሸንፈዋል። ሊንክ ይጫኑ።"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["scam_category"] == "FINANCIAL"
    assert b["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")


def test_text_english_job_scam():
    r = client.post("/api/v1/scan-text", json={
        "text": "We are hiring! Work from home, earn 25,000 birr/week. Send CV to t.me/jobs_et"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["scam_category"] == "JOB_SCAM"


def test_text_english_visa_scam():
    r = client.post("/api/v1/scan-text", json={
        "text": "Visa opportunity to Canada! Pay only the visa fee. Guaranteed visa."
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["scam_category"] == "VISA_SCHOLARSHIP"


def test_text_telegram_takeover_amharic():
    r = client.post("/api/v1/scan-text", json={
        "text": "የቴሌግራም አካውንትዎ ሊዘጋ ነው! የማረጋገጫ ኮድ ይላኩ።"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["scam_category"] == "TELEGRAM_ACCOUNT_TAKEOVER"


def test_text_afan_oromo_job_scam():
    r = client.post("/api/v1/scan-text", json={
        "text": "Carraa hojii! Mana keessaa hojjedhu, torbanitti qarshii 25,000 argadhu. CV ergi."
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["language_detected"] == "afan_oromo"
    assert b["scam_category"] == "JOB_SCAM"


def test_text_benign_amharic():
    r = client.post("/api/v1/scan-text", json={
        "text": "ሰላም ወዳጄ፤ ዛሬ ምሳ እንገናኝ?"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is False


def test_text_benign_english():
    r = client.post("/api/v1/scan-text", json={
        "text": "Hi, are we still meeting for lunch tomorrow?"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is False


# ------------------------------- URL ----------------------------------------

def test_url_typosquat_telebirr():
    r = client.post("/api/v1/scan-url", json={
        "url": "http://telebirr-login.verify-account.xyz/secure"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["impersonated_brand"] == "telebirr"
    assert b["brand_sector"] == "bank"
    assert b["scam_category"] == "FINANCIAL"


def test_url_telegram_lookalike():
    r = client.post("/api/v1/scan-url", json={
        "url": "http://telegram-verify.xyz/account"
    })
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert b["impersonated_brand"] == "telegram"
    assert b["scam_category"] == "TELEGRAM_ACCOUNT_TAKEOVER"


def test_url_shortener():
    r = client.post("/api/v1/scan-url", json={"url": "http://bit.ly/abc123"})
    assert r.status_code == 200
    b = r.json()
    assert any(reason["code"] == "SHORTENER" for reason in b["reasons"])


def test_url_legit():
    r = client.post("/api/v1/scan-url", json={"url": "https://telebirr.et/login"})
    assert r.status_code == 200
    b = r.json()
    assert b["impersonated_brand"] is None
    assert b["risk_level"] == "LOW"


def test_url_ip_host():
    r = client.post("/api/v1/scan-url", json={"url": "http://192.168.1.10/telebirr/login"})
    assert r.status_code == 200
    b = r.json()
    assert b["is_phishing"] is True
    assert any(reason["code"] == "IP_HOST" for reason in b["reasons"])


# ----------------------------- Validation -----------------------------------

def test_validation_empty_text():
    r = client.post("/api/v1/scan-text", json={"text": ""})
    assert r.status_code == 422