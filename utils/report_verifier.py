"""
MediNova Official Clinical Report Verification & Cryptographic Gateway
======================================================================
Provides durable, HIPAA-aligned verification registry for all generated
patient screening reports. Backed by the application's persistent SQLite core.
"""

import os
import hmac
import hashlib
import sqlite3
import datetime
import json
import logging
from io import BytesIO
from typing import Dict, Any, Optional

logger = logging.getLogger("medinova.report_verifier")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "medinova_appointments.db")
def get_signing_secret() -> str:
    """Retrieve report signing secret dynamically, failing closed in production if unset."""
    secret = os.environ.get("REPORT_SIGNING_SECRET") or os.environ.get("SECRET_KEY")
    is_prod = (
        os.environ.get("FLASK_ENV") == "production"
        or os.environ.get("ENV") == "production"
        or os.environ.get("RENDER") == "true"
        or os.environ.get("PRODUCTION", "").lower() in ["true", "1", "yes"]
    )
    if not secret:
        if is_prod:
            raise RuntimeError("CRITICAL SECURITY ERROR: REPORT_SIGNING_SECRET or SECRET_KEY must be configured in production.")
        secret = "medinova-clinical-report-seal-secret-2026"
    elif is_prod and secret in ("medinova-clinical-secret-key-2026", "medinova_clinical_report_seal_secret_2026"):
        raise RuntimeError("CRITICAL SECURITY ERROR: Default report signing secret cannot be used in production.")
    return secret


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_report_verifier_db():
    """Initializes the report_verifications table in medinova_appointments.db."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS report_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module_name TEXT NOT NULL,
                condition_title TEXT NOT NULL,
                prediction_text TEXT NOT NULL,
                is_positive INTEGER NOT NULL,
                confidence_score REAL,
                is_scanner_extracted INTEGER DEFAULT 0,
                summary_json TEXT,
                verification_hash TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rep_id ON report_verifications (report_id)")
        conn.commit()
    except Exception as e:
        logger.error(f"[VERIFIER DB] Init failed: {e}")
    finally:
        conn.close()


def generate_verification_hash(report_id: str, module_name: str, timestamp_str: str) -> str:
    """Computes HMAC-SHA256 integrity seal."""
    msg = f"{report_id}|{module_name}|{timestamp_str}".encode("utf-8")
    secret = get_signing_secret()
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:24].upper()


def register_report(
    report_id: str,
    module_name: str,
    condition_title: str,
    prediction_text: str,
    is_positive: bool,
    confidence_score: float,
    is_scanner_extracted: bool = False,
    summary_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Registers a report verification record.
    Returns verification_hash.
    """
    init_report_verifier_db()
    conn = get_db()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    v_hash = generate_verification_hash(report_id, module_name, now_str)

    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO report_verifications (
                report_id, created_at, module_name, condition_title,
                prediction_text, is_positive, confidence_score,
                is_scanner_extracted, summary_json, verification_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            now_str,
            module_name,
            condition_title,
            prediction_text,
            1 if is_positive else 0,
            float(confidence_score) if confidence_score is not None else None,
            1 if is_scanner_extracted else 0,
            json.dumps(summary_data or {}),
            v_hash
        ))
        conn.commit()
        return v_hash
    except Exception as e:
        logger.error(f"[VERIFIER DB] Registration error for {report_id}: {e}")
        return v_hash
    finally:
        conn.close()


def get_verified_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves verified report record by report_id (case-insensitive and trimmed)."""
    if not report_id:
        return None
    init_report_verifier_db()
    conn = get_db()
    clean_id = report_id.strip()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM report_verifications WHERE UPPER(report_id) = UPPER(?)", (clean_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    except Exception as e:
        logger.error(f"[VERIFIER DB] Fetch error for {report_id}: {e}")
        return None
    finally:
        conn.close()


def generate_qr_code_image(report_id: str, base_url: str) -> bytes:
    """
    Generates a PNG QR code for the public verification link.
    Guarantees no external third-party image dependencies.
    """
    import qrcode
    clean_base = base_url.rstrip("/")
    verification_url = f"{clean_base}/verify-report/{report_id}"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(verification_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#0891b2", back_color="#ffffff")
    buf = BytesIO()
    try:
        img.save(buf)
    except TypeError:
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, "PNG")
    return buf.getvalue()
