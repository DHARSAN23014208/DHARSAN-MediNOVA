"""
MediNova Calendar Integration Service
====================================
Provides:
1. One-click Google Calendar web event URL generator.
2. Standard RFC 5545 iCalendar (.ics) file generator.
Handles time slots, hospital location, consultation mode, and instructions.
"""

import urllib.parse
import datetime
from typing import Dict, Any, Tuple


TIME_SLOT_RANGES = {
    "09:00 - 10:00 AM": ("09:00:00", "10:00:00"),
    "10:00 - 11:00 AM": ("10:00:00", "11:00:00"),
    "11:00 - 12:00 PM": ("11:00:00", "12:00:00"),
    "02:00 - 03:00 PM": ("14:00:00", "15:00:00"),
    "03:00 - 04:00 PM": ("15:00:00", "16:00:00"),
    "04:00 - 05:00 PM": ("16:00:00", "17:00:00"),
}


def parse_appointment_datetimes(date_str: str, time_slot_str: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """Parse appointment date and time slot into start and end datetimes."""
    try:
        base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        base_date = datetime.datetime.now().date() + datetime.timedelta(days=1)

    start_time_str, end_time_str = TIME_SLOT_RANGES.get(time_slot_str, ("10:00:00", "11:00:00"))
    start_t = datetime.datetime.strptime(start_time_str, "%H:%M:%S").time()
    end_t = datetime.datetime.strptime(end_time_str, "%H:%M:%S").time()

    dt_start = datetime.datetime.combine(base_date, start_t)
    dt_end = datetime.datetime.combine(base_date, end_t)
    return dt_start, dt_end


def generate_google_calendar_url(appointment: Dict[str, Any]) -> str:
    """Generate a 1-click Google Calendar event addition URL."""
    dt_start, dt_end = parse_appointment_datetimes(
        appointment.get("preferred_date", ""),
        appointment.get("preferred_time_slot", "")
    )

    fmt = "%Y%m%dT%H%M%S"
    dates_param = f"{dt_start.strftime(fmt)}/{dt_end.strftime(fmt)}"

    cond_name = appointment.get("condition_name", "Specialist Consultation")
    hosp_name = appointment.get("hospital_name", "Medical Centre")
    ref_id = appointment.get("reference_id", "")
    mode = "In-Person Consultation" if appointment.get("consultation_mode") == "in_person" else "Telemedicine Consultation"

    title = f"MediNova Consultation: {cond_name} ({ref_id})"
    location = appointment.get("hospital_address", hosp_name)
    if appointment.get("consultation_mode") == "telemedicine":
        location = "Telemedicine Video Consultation (Link to be provided upon hospital confirmation)"

    desc_lines = [
        f"MediNova Priority Specialist Appointment Request",
        f"Reference ID: {ref_id}",
        f"Patient: {appointment.get('patient_name', '')}",
        f"Diagnostic Condition: {cond_name}",
        f"Facility: {hosp_name}",
        f"Address: {appointment.get('hospital_address', '')}",
        f"Mode: {mode}",
        f"Preferred Slot: {appointment.get('preferred_time_slot', '')}",
        f"Status: {appointment.get('status', 'Awaiting Hospital Confirmation')}",
        "",
        "IMPORTANT NOTICE:",
        "This appointment is currently Awaiting Hospital Confirmation. Please bring your MediNova diagnostic summary slip to your visit."
    ]
    if appointment.get("directions_url"):
        desc_lines.append(f"Directions: {appointment.get('directions_url')}")

    description = "\n".join(desc_lines)

    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": dates_param,
        "details": description,
        "location": location
    }
    return f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(params)}"


def generate_ics_content(appointment: Dict[str, Any]) -> str:
    """Generate RFC 5545 standard .ics file string."""
    dt_start, dt_end = parse_appointment_datetimes(
        appointment.get("preferred_date", ""),
        appointment.get("preferred_time_slot", "")
    )

    fmt = "%Y%m%dT%H%M%S"
    dt_stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ref_id = appointment.get("reference_id", "APT-EVENT")
    cond_name = appointment.get("condition_name", "Specialist Consultation")
    hosp_name = appointment.get("hospital_name", "Medical Centre")
    address = appointment.get("hospital_address", hosp_name)
    mode = "In-Person Consultation" if appointment.get("consultation_mode") == "in_person" else "Telemedicine Consultation"

    description = (
        f"MediNova Medical Appointment\\n"
        f"Reference ID: {ref_id}\\n"
        f"Patient: {appointment.get('patient_name')}\\n"
        f"Condition: {cond_name}\\n"
        f"Hospital: {hosp_name}\\n"
        f"Mode: {mode}\\n"
        f"Status: {appointment.get('status')}\\n"
        f"Note: Appointment requested through MediNova Clinical Discovery."
    )

    ics_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//MediNova Clinical AI//Appointment System 4.0//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{ref_id}@medinova.ai",
        f"DTSTAMP:{dt_stamp}",
        f"DTSTART:{dt_start.strftime(fmt)}",
        f"DTEND:{dt_end.strftime(fmt)}",
        f"SUMMARY:MediNova Consultation: {cond_name} - {hosp_name}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{address}",
        "STATUS:TENTATIVE",
        "TRANSP:OPAQUE",
        "END:VEVENT",
        "END:VCALENDAR"
    ]
    return "\r\n".join(ics_lines)
