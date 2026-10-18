"""
MediNova Persistent Appointment Database & Lifecycle Management
==============================================================
Production-grade SQLite database model for patient consultation requests.
Features:
- Parameterized queries for SQL injection prevention.
- Cryptographically secure unique Reference ID generation (e.g. APT-2026-AB12CD).
- Meaningful medical status workflow and timestamped audit history.
- Patient isolation: Lookup requires Reference ID + matching Phone or Email.
- Authenticated administrator/clinic coordinator access.
- Thread-safe connection handling.
"""

import os
import json
import time
import sqlite3
import secrets
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("medinova.appointment_db")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "medinova_appointments.db")

VALID_STATUSES = [
    "Request Submitted",
    "Awaiting Hospital Confirmation",
    "Confirmed by Hospital",
    "Rescheduling Required",
    "Cancelled",
    "Completed",
    "Rejected"
]


def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the appointment database tables and indices if not existing."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id TEXT UNIQUE NOT NULL,
            access_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            patient_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            condition_name TEXT NOT NULL,
            clinical_notes TEXT,
            hospital_id TEXT,
            hospital_name TEXT NOT NULL,
            facility_type TEXT,
            hospital_address TEXT NOT NULL,
            hospital_website TEXT,
            directions_url TEXT,
            distance_km REAL,
            specialty_label TEXT,
            consultation_mode TEXT NOT NULL,
            preferred_date TEXT NOT NULL,
            preferred_time_slot TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Awaiting Hospital Confirmation',
            notification_status TEXT DEFAULT 'Pending',
            is_urgent INTEGER DEFAULT 0,
            consent_given INTEGER NOT NULL DEFAULT 1,
            admin_notes TEXT,
            status_history TEXT NOT NULL
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_apt_ref ON appointments(reference_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_apt_email ON appointments(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_apt_phone ON appointments(phone);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_apt_status ON appointments(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_apt_created ON appointments(created_at);")
        conn.commit()
        logger.info(f"[DB INIT] SQLite appointment database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"[DB INIT ERROR] Failed to initialize SQLite appointment table: {e}")
        raise
    finally:
        conn.close()


def generate_reference_id() -> str:
    """
    Generate a cryptographically secure, unique, human-readable medical reference ID.
    Format: APT-2026-XXXXXX (alphanumeric uppercase)
    """
    year = datetime.datetime.now().year
    random_part = secrets.token_hex(3).upper()  # 6 uppercase hex characters
    return f"APT-{year}-{random_part}"


def create_appointment(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate and insert a new patient consultation appointment request into the database.
    Initial status is strictly 'Awaiting Hospital Confirmation'.
    """
    required_fields = [
        "patient_name", "phone", "email", "condition_id", "condition_name",
        "hospital_name", "hospital_address", "consultation_mode",
        "preferred_date", "preferred_time_slot"
    ]
    for field in required_fields:
        if not str(data.get(field, "")).strip():
            return False, {}, f"Required field '{field}' is missing or empty."

    # Validate email
    email = str(data["email"]).strip().lower()
    if "@" not in email or "." not in email:
        return False, {}, "A valid email address is required for appointment dispatch."

    # Validate phone
    phone = str(data["phone"]).strip()
    if len(phone) < 7:
        return False, {}, "A valid phone number is required for doctor coordination."

    # Validate consultation mode
    mode = str(data["consultation_mode"]).strip().lower()
    if mode not in ["in_person", "telemedicine"]:
        mode = "in_person"

    # Validate preferred date (cannot be in the past)
    pref_date_str = str(data["preferred_date"]).strip()
    try:
        pref_date = datetime.datetime.strptime(pref_date_str, "%Y-%m-%d").date()
        today = datetime.datetime.now().date()
        if pref_date < today:
            return False, {}, "Preferred appointment date cannot be in the past."
    except ValueError:
        return False, {}, "Preferred date must be in YYYY-MM-DD format."

    # Ensure consent
    if not data.get("consent_given", True):
        return False, {}, "Patient consent is required to process appointment request."

    ref_id = generate_reference_id()
    access_token = secrets.token_urlsafe(24)
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    initial_status = "Awaiting Hospital Confirmation"
    history = [
        {
            "status": initial_status,
            "timestamp": now_iso,
            "actor": "Patient Request (Online Portal)",
            "note": "Initial consultation request submitted by patient. Awaiting clinical coordinator / hospital confirmation."
        }
    ]

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO appointments (
                reference_id, access_token, patient_name, phone, email,
                condition_id, condition_name, clinical_notes, hospital_id,
                hospital_name, facility_type, hospital_address, hospital_website,
                directions_url, distance_km, specialty_label, consultation_mode,
                preferred_date, preferred_time_slot, status, notification_status,
                is_urgent, consent_given, admin_notes, status_history
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
        """, (
            ref_id,
            access_token,
            data["patient_name"].strip(),
            phone,
            email,
            data["condition_id"].strip().lower(),
            data["condition_name"].strip(),
            data.get("clinical_notes", "").strip(),
            data.get("hospital_id", ""),
            data["hospital_name"].strip(),
            data.get("facility_type", "Healthcare Facility"),
            data["hospital_address"].strip(),
            data.get("hospital_website", "").strip(),
            data.get("directions_url", "").strip(),
            data.get("distance_km"),
            data.get("specialty_label", "Specialty Matched"),
            mode,
            pref_date_str,
            data["preferred_time_slot"].strip(),
            initial_status,
            "Pending",
            1 if data.get("is_urgent") else 0,
            1,
            None,
            json.dumps(history)
        ))
        conn.commit()
        record_id = cursor.lastrowid

        created_record = {
            "id": record_id,
            "reference_id": ref_id,
            "access_token": access_token,
            "created_at": now_iso,
            "patient_name": data["patient_name"].strip(),
            "phone": phone,
            "email": email,
            "condition_id": data["condition_id"].strip().lower(),
            "condition_name": data["condition_name"].strip(),
            "clinical_notes": data.get("clinical_notes", "").strip(),
            "hospital_name": data["hospital_name"].strip(),
            "facility_type": data.get("facility_type", "Healthcare Facility"),
            "hospital_address": data["hospital_address"].strip(),
            "hospital_website": data.get("hospital_website", "").strip(),
            "directions_url": data.get("directions_url", "").strip(),
            "distance_km": data.get("distance_km"),
            "specialty_label": data.get("specialty_label", "Specialty Matched"),
            "consultation_mode": mode,
            "preferred_date": pref_date_str,
            "preferred_time_slot": data["preferred_time_slot"].strip(),
            "status": initial_status,
            "notification_status": "Pending",
            "is_urgent": 1 if data.get("is_urgent") else 0,
            "status_history": history
        }
        return True, created_record, None
    except Exception as e:
        logger.error(f"[DB INSERT ERROR] Failed to create appointment: {e}")
        conn.rollback()
        return False, {}, str(e)
    finally:
        conn.close()


def get_appointment_by_ref(reference_id: str, verification_contact: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve appointment record by unique reference ID.
    If verification_contact is provided (phone or email), verifies patient ownership
    to enforce strict patient record isolation.
    """
    reference_id = (reference_id or "").strip().upper()
    if not reference_id:
        return None

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM appointments WHERE reference_id = ?", (reference_id,))
        row = cursor.fetchone()
        if not row:
            return None

        record = dict(row)
        try:
            record["status_history"] = json.loads(record.get("status_history") or "[]")
        except Exception:
            record["status_history"] = []

        if verification_contact:
            verify_clean = verification_contact.strip().lower()
            rec_email = record.get("email", "").lower()
            rec_phone = record.get("phone", "").replace(" ", "").replace("-", "")
            verify_phone = verify_clean.replace(" ", "").replace("-", "")

            # Match either email or phone
            if verify_clean != rec_email and (len(verify_phone) < 5 or verify_phone not in rec_phone):
                logger.warning(f"[SECURITY ISOLATION] Access blocked for ref {reference_id}: contact mismatch")
                return None

        return record
    finally:
        conn.close()


def get_appointment_by_token(access_token: str) -> Optional[Dict[str, Any]]:
    """Retrieve appointment using patient's secret access token (magic link)."""
    if not access_token:
        return None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM appointments WHERE access_token = ?", (access_token,))
        row = cursor.fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["status_history"] = json.loads(record.get("status_history") or "[]")
        except Exception:
            record["status_history"] = []
        return record
    finally:
        conn.close()


def list_appointments_admin(status_filter: Optional[str] = None, condition_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve appointment records for authenticated clinic administrators / developer.
    Supports filtering by status and diagnostic condition.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM appointments WHERE 1=1"
        params: List[Any] = []

        if status_filter and status_filter in VALID_STATUSES:
            query += " AND status = ?"
            params.append(status_filter)

        if condition_filter:
            query += " AND condition_id = ?"
            params.append(condition_filter.strip().lower())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            item = dict(r)
            try:
                item["status_history"] = json.loads(item.get("status_history") or "[]")
            except Exception:
                item["status_history"] = []
            results.append(item)
        return results
    finally:
        conn.close()


def update_appointment_status(
    reference_id: str,
    new_status: str,
    actor_name: str,
    admin_notes: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Perform an authorized status transition on an appointment record with an audit entry.
    """
    reference_id = (reference_id or "").strip().upper()
    if new_status not in VALID_STATUSES:
        return False, f"Invalid status '{new_status}'. Allowed: {', '.join(VALID_STATUSES)}"

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT status, status_history, admin_notes FROM appointments WHERE reference_id = ?", (reference_id,))
        row = cursor.fetchone()
        if not row:
            return False, f"Appointment reference '{reference_id}' not found."

        old_status = row["status"]
        history_str = row["status_history"] or "[]"
        try:
            history = json.loads(history_str)
        except Exception:
            history = []

        now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({
            "status": new_status,
            "previous_status": old_status,
            "timestamp": now_iso,
            "actor": actor_name,
            "note": admin_notes or f"Status updated from '{old_status}' to '{new_status}'."
        })

        cursor.execute("""
            UPDATE appointments
            SET status = ?, status_history = ?, admin_notes = ?
            WHERE reference_id = ?
        """, (new_status, json.dumps(history), admin_notes, reference_id))
        conn.commit()
        logger.info(f"[STATUS TRANSITION] Appointment {reference_id}: '{old_status}' -> '{new_status}' by {actor_name}")
        return True, None
    except Exception as e:
        logger.error(f"[STATUS UPDATE ERROR] Failed for {reference_id}: {e}")
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()


def cancel_appointment_patient(reference_id: str, verification_contact: str, reason: Optional[str] = None) -> Tuple[bool, str]:
    """
    Allows a patient to cancel an appointment if it is in an eligible pending state.
    Requires reference ID and matching phone/email for access control.
    """
    record = get_appointment_by_ref(reference_id, verification_contact)
    if not record:
        return False, "Appointment not found or contact verification failed."

    current_status = record.get("status")
    if current_status in ["Completed", "Cancelled", "Rejected"]:
        return False, f"Cannot cancel appointment currently in '{current_status}' state."

    success, err = update_appointment_status(
        reference_id=reference_id,
        new_status="Cancelled",
        actor_name="Patient (Self-Service Online Portal)",
        admin_notes=reason or "Patient requested cancellation via self-service portal."
    )
    if success:
        return True, "Your appointment request has been successfully cancelled."
    return False, err or "Failed to cancel appointment."


def update_notification_status(reference_id: str, new_notification_status: str):
    """Update notification dispatch status (Delivered, Pending, Failed)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE appointments
            SET notification_status = ?
            WHERE reference_id = ?
        """, (new_notification_status, reference_id.strip().upper()))
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to update notification status for {reference_id}: {e}")
    finally:
        conn.close()


# Automatically initialize schema on module import
init_db()
