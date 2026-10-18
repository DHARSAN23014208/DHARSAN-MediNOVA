"""
MediNova Appointment Notification & Dispatch Service
===================================================
Dispatches appointment notifications to:
1. Authorized clinic administrator / developer email via configured SMTP.
2. Verified hospital contact/coordination desk when configured.
3. Updates appointment database record notification status (Delivered, Pending, Failed).
"""

import os
import logging
from typing import Dict, Any, Tuple
from utils.appointment_db import update_notification_status

logger = logging.getLogger("medinova.appointment_notifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def notify_appointment_request(appointment: Dict[str, Any], send_email_func=None) -> Tuple[bool, str]:
    """
    Dispatch a priority consultation alert to the authorized clinic care coordinator / developer.
    Records delivery outcome in the appointment database record.
    """
    ref_id = appointment.get("reference_id", "")
    patient_name = appointment.get("patient_name", "Patient")
    cond_name = appointment.get("condition_name", "Condition")
    hosp_name = appointment.get("hospital_name", "Hospital")
    mode = "In-Person Consultation" if appointment.get("consultation_mode") == "in_person" else "Telemedicine Consultation"
    pref_date = appointment.get("preferred_date", "")
    pref_slot = appointment.get("preferred_time_slot", "")
    phone = appointment.get("phone", "")
    email = appointment.get("email", "")
    notes = appointment.get("clinical_notes", "None provided")
    is_urgent = bool(appointment.get("is_urgent"))

    subject = f"[MediNova Alert] {'🚨 URGENT ' if is_urgent else ''}New Appointment Request: {ref_id} ({cond_name})"

    urgency_badge = """
    <div style="background: #ef4444; color: #fff; padding: 8px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 16px;">
        🚨 CLINICAL NOTICE: Patient symptoms or condition indicate elevated urgency. Expedited hospital triage recommended.
    </div>
    """ if is_urgent else ""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; background: #0b1120; color: #f8fafc; padding: 24px; border-radius: 8px; max-width: 600px; margin: 0 auto;">
        <div style="border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 16px;">
            <h2 style="color: #38bdf8; margin: 0;">MediNova Priority Specialist Appointment Request</h2>
            <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;">Intelligent Hospital Discovery & Scheduling Gateway</p>
        </div>

        {urgency_badge}

        <table style="width: 100%; border-collapse: collapse; margin-bottom: 16px;">
            <tr>
                <td style="padding: 6px 0; color: #94a3b8; width: 40%;">Reference ID:</td>
                <td style="padding: 6px 0; font-weight: bold; color: #38bdf8; font-family: monospace; font-size: 16px;">{ref_id}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Patient Name:</td>
                <td style="padding: 6px 0; font-weight: bold; color: #f8fafc;">{patient_name}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Phone:</td>
                <td style="padding: 6px 0; color: #f8fafc;">{phone}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Email:</td>
                <td style="padding: 6px 0; color: #f8fafc;">{email}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Diagnostic Module:</td>
                <td style="padding: 6px 0; font-weight: bold; color: #34d399;">{cond_name}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Selected Facility:</td>
                <td style="padding: 6px 0; font-weight: bold; color: #f8fafc;">{hosp_name}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Address:</td>
                <td style="padding: 6px 0; color: #94a3b8; font-size: 13px;">{appointment.get('hospital_address', '')}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Consultation Mode:</td>
                <td style="padding: 6px 0; color: #f8fafc;">{mode}</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Requested Date / Slot:</td>
                <td style="padding: 6px 0; color: #f8fafc;">{pref_date} ({pref_slot})</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; color: #94a3b8;">Current Status:</td>
                <td style="padding: 6px 0; font-weight: bold; color: #f59e0b;">Awaiting Hospital Confirmation</td>
            </tr>
        </table>

        <div style="background: #1e293b; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
            <strong style="color: #94a3b8; font-size: 12px; text-transform: uppercase;">Patient Clinical Notes:</strong>
            <p style="margin: 6px 0 0 0; font-size: 13px; color: #f1f5f9; line-height: 1.4;">{notes}</p>
        </div>

        <p style="font-size: 12px; color: #64748b; margin-top: 20px;">
            This is an automated dispatch from MediNova Hospital Discovery. The status must remain "Awaiting Hospital Confirmation" until validated by clinic scheduling.
        </p>
    </div>
    """

    if callable(send_email_func):
        try:
            sent, msg = send_email_func(subject, body_html)
            if sent:
                update_notification_status(ref_id, "Delivered (Clinic Coordinator)")
                return True, "Notification successfully dispatched to clinic coordinator."
            else:
                update_notification_status(ref_id, "Pending (SMTP Unconfigured)")
                return False, f"Notification recorded. SMTP notice: {msg}"
        except Exception as e:
            logger.error(f"[NOTIFIER ERROR] Failed to send email for {ref_id}: {e}")
            update_notification_status(ref_id, f"Failed: {e}")
            return False, str(e)

    update_notification_status(ref_id, "Pending (Logged to Database)")
    return True, "Appointment recorded in database. Awaiting hospital coordinator processing."
