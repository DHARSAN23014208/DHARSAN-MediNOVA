"""
MediNova Private Google Sheets Patient Diagnostic Records Logger
================================================================
Replaces local Excel/CSV logging with private, silent, cloud-only Google Sheets logging.
Target Spreadsheet: MediNova_Patient_Records
Strict Developer Access: dharshankumarkumar3@gmail.com

Architecture:
- Background daemon worker thread with bounded task queue for ZERO prediction latency.
- Thread-safe state management with threading.Lock().
- Bounded retries with exponential backoff for Google API rate-limits and network hiccups.
- Fault-tolerant: Exceptions are logged; patient diagnostic responses NEVER fail.
- Automatic spreadsheet tab & header initialization across all 8 worksheets:
    1. Master_Log
    2. Diabetes
    3. Heart_Disease
    4. Kidney_Disease
    5. Liver_Disease
    6. Breast_Cancer
    7. Malaria
    8. Pneumonia
"""

import os
import json
import time
import queue
import uuid
import datetime
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger("medinova.google_sheets_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Authorized developer email (Strictly dharshankumarkumar3@gmail.com)
AUTHORIZED_DEV_EMAIL = os.environ.get("AUTHORIZED_DEV_EMAIL", "dharshankumarkumar3@gmail.com").strip().lower()

# Google Sheets Spreadsheet Name & Configuration
SPREADSHEET_NAME = "MediNova_Patient_Records"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Worksheet Tab Names
SHEET_MASTER = "Master_Log"
SHEET_DIABETES = "Diabetes"
SHEET_HEART = "Heart_Disease"
SHEET_KIDNEY = "Kidney_Disease"
SHEET_LIVER = "Liver_Disease"
SHEET_BREAST = "Breast_Cancer"
SHEET_MALARIA = "Malaria"
SHEET_PNEUMONIA = "Pneumonia"

ALL_SHEET_NAMES = [
    SHEET_MASTER,
    SHEET_DIABETES,
    SHEET_HEART,
    SHEET_KIDNEY,
    SHEET_LIVER,
    SHEET_BREAST,
    SHEET_MALARIA,
    SHEET_PNEUMONIA
]

MODULE_TO_SHEET = {
    "diabetes": SHEET_DIABETES,
    "heart": SHEET_HEART,
    "heart_disease": SHEET_HEART,
    "kidney": SHEET_KIDNEY,
    "kidney_disease": SHEET_KIDNEY,
    "liver": SHEET_LIVER,
    "liver_disease": SHEET_LIVER,
    "breast_cancer": SHEET_BREAST,
    "breast": SHEET_BREAST,
    "malaria": SHEET_MALARIA,
    "pneumonia": SHEET_PNEUMONIA,
}

# Canonical Column Definitions (Clean separate column per clinical parameter)
SHEET_COLUMNS: Dict[str, List[str]] = {
    SHEET_MASTER: [
        "Record_ID", "Timestamp", "Disease_Module", "Patient_Summary",
        "Image_or_Key_Feature", "Raw_Prediction", "Prediction_Status",
        "Confidence_Score", "Notes"
    ],
    SHEET_DIABETES: [
        "Record_ID", "Timestamp", "Disease_Module",
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age",
        "gender", "smoking_history", "hypertension", "heart_disease",
        "HbA1c_level", "blood_glucose_level",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_HEART: [
        "Record_ID", "Timestamp", "Disease_Module",
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "bp", "ekg", "max_hr",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_KIDNEY: [
        "Record_ID", "Timestamp", "Disease_Module",
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_LIVER: [
        "Record_ID", "Timestamp", "Disease_Module",
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
        "Albumin_and_Globulin_Ratio",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_BREAST: [
        "Record_ID", "Timestamp", "Disease_Module",
        "mean_radius", "mean_texture", "mean_perimeter",
        "mean_area", "mean_smoothness",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_MALARIA: [
        "Record_ID", "Timestamp", "Disease_Module",
        "Image_Filename", "Original_Resolution", "Input_Shape", "Model_Used",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ],
    SHEET_PNEUMONIA: [
        "Record_ID", "Timestamp", "Disease_Module",
        "Image_Filename", "Original_Resolution", "Input_Shape", "Model_Used",
        "Raw_Prediction", "Prediction_Status", "Confidence_Score"
    ]
}

# In-memory logging queue for silent asynchronous Google Sheets worker
MAX_QUEUE_SIZE = 2000
_log_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
_client_lock = threading.Lock()
_sheets_service = None
_initialized_sheets_cache = set()
_last_status = {
    "total_logged": 0,
    "last_sync_time": None,
    "last_error": None,
    "is_connected": False
}


def _get_credentials():
    """
    Authenticate service account credentials from:
    1. GOOGLE_CREDENTIALS_JSON (raw JSON string in environment)
    2. GOOGLE_APPLICATION_CREDENTIALS (file path in environment)
    3. service_account.json in project root
    """
    # 1. Environment variable containing JSON string
    cred_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if cred_json_str:
        try:
            info = json.loads(cred_json_str)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            logger.error(f"[Google Auth] Failed parsing GOOGLE_CREDENTIALS_JSON: {e}")

    # 2. Environment variable pointing to file path
    cred_file_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_file_path and os.path.exists(cred_file_path):
        try:
            return service_account.Credentials.from_service_account_file(cred_file_path, scopes=SCOPES)
        except Exception as e:
            logger.error(f"[Google Auth] Failed loading credentials from {cred_file_path}: {e}")

    # 3. Local file in project directory (for development)
    local_service_file = os.path.join(BASE_DIR, "service_account.json")
    if os.path.exists(local_service_file):
        try:
            return service_account.Credentials.from_service_account_file(local_service_file, scopes=SCOPES)
        except Exception as e:
            logger.error(f"[Google Auth] Failed loading local service_account.json: {e}")

    return None


def get_spreadsheet_id() -> Optional[str]:
    """Retrieve target Google Spreadsheet ID from environment variable."""
    return os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip() or None


def get_sheets_service():
    """
    Thread-safe lazy initialization of the Google Sheets API client.
    Returns the service instance or None if unconfigured.
    """
    global _sheets_service
    with _client_lock:
        if _sheets_service is not None:
            return _sheets_service

        creds = _get_credentials()
        if not creds:
            logger.warning(
                "[Google Sheets] Credentials not configured. "
                "Set GOOGLE_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS to enable cloud patient logging."
            )
            return None

        try:
            _sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
            logger.info("[Google Sheets] Service account client authenticated successfully.")
            return _sheets_service
        except Exception as e:
            logger.error(f"[Google Sheets] Failed to initialize Google Sheets client: {e}")
            return None


def _ensure_spreadsheet_sheets_and_headers(service, spreadsheet_id: str) -> bool:
    """
    Ensures all 8 worksheet tabs exist and have their stable headers in row 1.
    Caches verified sheets to avoid repetitive API calls.
    """
    global _initialized_sheets_cache

    try:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        existing_sheet_map = {
            s["properties"]["title"]: s["properties"]["sheetId"]
            for s in meta.get("sheets", [])
        }

        # 1. Add missing worksheets
        missing_sheets = [name for name in ALL_SHEET_NAMES if name not in existing_sheet_map]
        if missing_sheets:
            requests = [
                {"addSheet": {"properties": {"title": sheet_name}}}
                for sheet_name in missing_sheets
            ]
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": requests}
            ).execute()
            logger.info(f"[Google Sheets] Created {len(missing_sheets)} missing worksheets: {missing_sheets}")

            # Re-fetch metadata for new sheet IDs
            meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            existing_sheet_map = {
                s["properties"]["title"]: s["properties"]["sheetId"]
                for s in meta.get("sheets", [])
            }

        # 2. Check and populate headers for uninitialized sheets
        sheets_to_check = [s for s in ALL_SHEET_NAMES if s not in _initialized_sheets_cache]
        if sheets_to_check:
            ranges = [f"'{name}'!A1:Z1" for name in sheets_to_check]
            val_res = service.spreadsheets().values().batchGet(
                spreadsheetId=spreadsheet_id,
                ranges=ranges
            ).execute()

            header_updates = []
            for item in val_res.get("valueRanges", []):
                r_title = item.get("range", "").split("!")[0].strip("'")
                current_values = item.get("values", [])
                expected_headers = SHEET_COLUMNS.get(r_title, [])

                if not current_values or not current_values[0]:
                    header_updates.append({
                        "range": f"'{r_title}'!A1",
                        "values": [expected_headers]
                    })
                _initialized_sheets_cache.add(r_title)

            if header_updates:
                service.spreadsheets().values().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={
                        "valueInputOption": "USER_ENTERED",
                        "data": header_updates
                    }
                ).execute()
                logger.info(f"[Google Sheets] Initialized headers for {len(header_updates)} sheets.")

        _last_status["is_connected"] = True
        return True

    except Exception as e:
        logger.error(f"[Google Sheets] Error validating spreadsheet structure: {e}")
        _last_status["last_error"] = str(e)
        _last_status["is_connected"] = False
        return False


def normalize_status(prediction_status: Any, is_positive: Optional[bool] = None) -> str:
    """
    Normalize status strictly to 'Positive' (disease detected) or 'Negative' (healthy/normal).
    """
    if is_positive is not None:
        return "Positive" if bool(is_positive) else "Negative"

    if isinstance(prediction_status, bool):
        return "Positive" if prediction_status else "Negative"

    if isinstance(prediction_status, (int, float)):
        return "Positive" if int(prediction_status) == 1 else "Negative"

    status_str = str(prediction_status).strip().lower()

    # Explicit negative check first (handles "uninfected", "benign", "absence", "normal", "healthy", etc.)
    negative_indicators = [
        "uninfected", "benign", "negative", "neg", "normal",
        "clear", "healthy", "absence", "low probability"
    ]
    if any(ind in status_str for ind in negative_indicators):
        return "Negative"

    positive_indicators = [
        "1", "true", "positive", "pos", "parasitized", "malignant",
        "detected", "high probability", "presence", "elevated", "disease", "infected"
    ]
    if any(ind in status_str for ind in positive_indicators):
        return "Positive"

    return "Negative"


def _format_confidence(confidence: Any) -> str:
    """Format confidence score to consistent percentage string."""
    if confidence is None:
        return "N/A"
    try:
        val = float(confidence)
        if 0.0 <= val <= 1.0 and val > 0:
            val = val * 100.0
        return f"{val:.2f}%"
    except (ValueError, TypeError):
        return str(confidence)


def _map_form_aliases(module_key: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps incoming clinical form keys to both prompt requirements and template inputs.
    Preserves valid zero values (e.g., pregnancies=0, oldpeak=0.0).
    """
    mapped = dict(form_data)

    if module_key in ["heart", "heart_disease"]:
        if "bp" in mapped and "trestbps" not in mapped:
            mapped["trestbps"] = mapped["bp"]
        if "trestbps" in mapped and "bp" not in mapped:
            mapped["bp"] = mapped["trestbps"]

        if "ekg" in mapped and "restecg" not in mapped:
            mapped["restecg"] = mapped["ekg"]
        if "restecg" in mapped and "ekg" not in mapped:
            mapped["ekg"] = mapped["restecg"]

        if "max_hr" in mapped and "thalach" not in mapped:
            mapped["thalach"] = mapped["max_hr"]
        if "thalach" in mapped and "max_hr" not in mapped:
            mapped["max_hr"] = mapped["thalach"]

    elif module_key in ["diabetes"]:
        if "blood_glucose_level" in mapped and "Glucose" not in mapped:
            mapped["Glucose"] = mapped["blood_glucose_level"]
        if "Glucose" in mapped and "blood_glucose_level" not in mapped:
            mapped["blood_glucose_level"] = mapped["Glucose"]

        if "age" in mapped and "Age" not in mapped:
            mapped["Age"] = mapped["age"]
        if "Age" in mapped and "age" not in mapped:
            mapped["age"] = mapped["Age"]

        if "bmi" in mapped and "BMI" not in mapped:
            mapped["BMI"] = mapped["bmi"]
        if "BMI" in mapped and "bmi" not in mapped:
            mapped["bmi"] = mapped["BMI"]

        if "BloodPressure" not in mapped and "bp" in mapped:
            mapped["BloodPressure"] = mapped["bp"]

    return mapped


def _clean_cell_value(val: Any) -> Any:
    """Ensure safe cell value formatting for Google Sheets values API."""
    if val is None:
        return ""
    if isinstance(val, (int, float, bool)):
        return val
    s = str(val).strip()
    return s


def _build_row_values(sheet_name: str, payload: Dict[str, Any]) -> List[Any]:
    """Build row vector matching worksheet columns in exact canonical order."""
    columns = SHEET_COLUMNS.get(sheet_name, list(payload.keys()))
    row = []
    for col in columns:
        val = payload.get(col, "")
        row.append(_clean_cell_value(val))
    return row


def _worker_loop():
    """
    Daemon worker thread processing the silent background logging queue.
    Appends records to Google Sheets without blocking patient diagnostic predictions.
    """
    logger.info("[Google Sheets Worker] Background logger thread started.")

    while True:
        task = _log_queue.get()
        if task is None:
            break

        module_sheet, module_row, master_row, record_id = task
        spreadsheet_id = get_spreadsheet_id()

        if not spreadsheet_id:
            logger.warning(
                f"[Google Sheets Worker] Logging skipped for record {record_id}: "
                f"GOOGLE_SHEETS_SPREADSHEET_ID not set. Prediction remains fully successful."
            )
            _log_queue.task_done()
            continue

        service = get_sheets_service()
        if not service:
            logger.warning(
                f"[Google Sheets Worker] Logging skipped for record {record_id}: "
                f"Google service account credentials unconfigured."
            )
            _log_queue.task_done()
            continue

        # Bounded Retries with Exponential Backoff
        max_retries = 3
        success = False
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                # 1. Ensure sheet tabs and headers exist
                _ensure_spreadsheet_sheets_and_headers(service, spreadsheet_id)

                # 2. Append to Disease Module Sheet
                body_mod = {"values": [module_row]}
                service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{module_sheet}'!A1",
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body_mod
                ).execute()

                # 3. Append to Master_Log Sheet
                body_master = {"values": [master_row]}
                service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{SHEET_MASTER}'!A1",
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body_master
                ).execute()

                success = True
                _last_status["total_logged"] += 1
                _last_status["last_sync_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _last_status["last_error"] = None
                _last_status["is_connected"] = True

                logger.info(f"[Google Sheets Sync] Successfully appended record {record_id} to '{module_sheet}' & '{SHEET_MASTER}'.")
                break

            except HttpError as http_err:
                last_exc = http_err
                status_code = http_err.resp.status if hasattr(http_err, "resp") else 0
                logger.warning(f"[Google Sheets Retry {attempt}/{max_retries}] HTTP {status_code} for record {record_id}: {http_err}")
                if status_code in [429, 500, 502, 503, 504] and attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    break
            except Exception as e:
                last_exc = e
                logger.warning(f"[Google Sheets Retry {attempt}/{max_retries}] Error logging {record_id}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    break

        if not success:
            _last_status["last_error"] = str(last_exc)
            logger.error(
                f"[Google Sheets Error] Permanent failure appending record {record_id} "
                f"after {max_retries} attempts: {last_exc}. (Patient response remained unaffected)."
            )

        _log_queue.task_done()


# Initialize daemon worker thread on import
_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="MediNova-GSheets-Worker")
_worker_thread.start()


def log_patient_prediction(
    module_name: str,
    form_data: Optional[Dict[str, Any]] = None,
    raw_prediction: Any = "",
    is_positive: Optional[bool] = None,
    confidence_score: Any = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Public logging interface for MediNova patient diagnostic routes.
    Queues prediction payloads asynchronously for background Google Sheets sync.
    Zero patient visibility & zero latency added to patient inference.

    Returns:
        (success: bool, record_id: str)
    """
    try:
        norm_module = module_name.strip().lower().replace("-", "_").replace(" ", "_")
        target_sheet_name = MODULE_TO_SHEET.get(norm_module)
        if not target_sheet_name:
            target_sheet_name = norm_module.title()

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        prefix_code = norm_module[:4].upper()
        record_id = f"MNV-{prefix_code}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

        status_str = normalize_status(raw_prediction, is_positive=is_positive)
        conf_str = _format_confidence(confidence_score)

        # 1. Build Disease Module Payload
        data_payload: Dict[str, Any] = {}
        if form_data:
            data_payload.update(_map_form_aliases(norm_module, form_data))
        if extra_metadata:
            data_payload.update(extra_metadata)

        data_payload["Record_ID"] = record_id
        data_payload["Timestamp"] = timestamp_str
        data_payload["Disease_Module"] = target_sheet_name
        data_payload["Raw_Prediction"] = str(raw_prediction)
        data_payload["Prediction_Status"] = status_str
        data_payload["Confidence_Score"] = conf_str

        module_row = _build_row_values(target_sheet_name, data_payload)

        # 2. Build Master_Log Summary Payload
        summary_parts = []
        image_or_key = data_payload.get("Image_Filename", "")
        for k in ["age", "Age", "Gender", "gender", "Glucose", "blood_glucose_level", "trestbps", "bp"]:
            if k in data_payload and data_payload[k] is not None and str(data_payload[k]).strip() != "":
                summary_parts.append(f"{k}: {data_payload[k]}")
        patient_summary_text = ", ".join(summary_parts) if summary_parts else (image_or_key or "Standard Clinical Assessment")

        master_payload = {
            "Record_ID": record_id,
            "Timestamp": timestamp_str,
            "Disease_Module": target_sheet_name,
            "Patient_Summary": patient_summary_text,
            "Image_or_Key_Feature": str(image_or_key or "N/A"),
            "Raw_Prediction": str(raw_prediction),
            "Prediction_Status": status_str,
            "Confidence_Score": conf_str,
            "Notes": f"Logged via {norm_module} route"
        }
        master_row = _build_row_values(SHEET_MASTER, master_payload)

        # 3. Enqueue for background worker
        try:
            _log_queue.put_nowait((target_sheet_name, module_row, master_row, record_id))
            logger.info(f"[Silent Log] Enqueued {target_sheet_name} record {record_id} ({status_str}) for Google Sheets.")
            return True, record_id
        except queue.Full:
            logger.warning(
                f"[Queue Saturated] Log queue reached capacity ({MAX_QUEUE_SIZE}). "
                f"Record {record_id} dropped to preserve patient response performance."
            )
            return False, "Queue saturated"

    except Exception as e:
        logger.error(f"[Google Sheets Logger Exception] Failed enqueuing prediction: {e}", exc_info=True)
        return False, str(e)


def flush_log_queue(timeout: float = 10.0) -> bool:
    """Block until all enqueued records have been processed by the worker (useful for tests/exit)."""
    start_time = time.time()
    while not _log_queue.empty():
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)
    return True


def get_sheets_status() -> Dict[str, Any]:
    """
    Returns Google Sheets synchronization status for the Developer Portal.
    No secrets or patient clinical parameters are exposed.
    """
    sp_id = get_spreadsheet_id()
    creds = _get_credentials()

    masked_id = f"{sp_id[:6]}...{sp_id[-4:]}" if sp_id and len(sp_id) > 12 else (sp_id or "Not Configured")
    direct_url = f"https://docs.google.com/spreadsheets/d/{sp_id}/edit" if sp_id else None

    return {
        "spreadsheet_name": SPREADSHEET_NAME,
        "spreadsheet_id": sp_id,
        "masked_id": masked_id,
        "direct_url": direct_url,
        "has_credentials": creds is not None,
        "authorized_developer": AUTHORIZED_DEV_EMAIL,
        "queue_pending": _log_queue.qsize(),
        "total_logged": _last_status["total_logged"],
        "last_sync_time": _last_status["last_sync_time"] or "Awaiting First Record",
        "last_error": _last_status["last_error"],
        "is_connected": _last_status["is_connected"],
        "sheet_tabs": ALL_SHEET_NAMES
    }
