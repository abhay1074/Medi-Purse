"""Email notifications: appointment confirmations, changes, reminders, portal invites.

Emails are queued in the `notifications` table and sent by a background worker,
so a slow mail server never blocks a page. The same worker sends reminders before
each scheduled appointment (REMINDER_HOURS, default 24 and 2 hours before).

Settings (.env):
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_TLS (1)
    APP_BASE_URL   link used in emails (default http://localhost:8000)
    REMINDER_HOURS comma-separated hours before the visit (default "24,2")

Without SMTP_HOST, emails are saved as .eml files in ./outbox (open them with
any mail program) and marked "saved" so the whole flow can be tried locally.
"""

from __future__ import annotations

import os
import smtplib
import ssl
import threading
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from html import escape
from pathlib import Path

from .db import ROOT, Appointment, Notification, Patient, SessionLocal

OUTBOX = ROOT / "outbox"
MAX_ATTEMPTS = 3
VISIT_MINUTES = 30


def base_url() -> str:
    return os.environ.get("APP_BASE_URL", "http://localhost:8000").rstrip("/")


def reminder_hours() -> list[int]:
    raw = os.environ.get("REMINDER_HOURS", "24,2")
    hours = sorted({int(x) for x in raw.split(",") if x.strip().isdigit() and int(x) > 0}, reverse=True)
    return hours or [24]


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def _fmt(when: datetime) -> str:
    return when.strftime("%A, %d %B %Y at %I:%M %p")


def _place(a: Appointment) -> str:
    c = a.patient.clinic
    parts = [c.label]
    if c.address:
        parts.append(c.address)
    if c.hospital and c.hospital.city:
        parts.append(c.hospital.city)
    return ", ".join(parts)


# ----------------------------------------------------------------------------
# message builders
# ----------------------------------------------------------------------------
def _appt_lines(a: Appointment) -> list[str]:
    lines = [f"When: {_fmt(a.scheduled_at)}", f"Where: {_place(a)}"]
    if a.doctor:
        lines.append(f"Doctor: {a.doctor.display}")
    if a.reason:
        lines.append(f"Reason: {a.reason}")
    lines.append(f"Patient ID: {a.patient.mrn}")
    return lines


def _footer(a: Appointment | None) -> list[str]:
    phone = a.patient.clinic.phone if a else ""
    out = ["", f"Manage your appointments online: {base_url()}/portal"]
    if phone:
        out.append(f"Or call the clinic: {phone}")
    out += ["", "MediPurse"]
    return out


def appointment_message(a: Appointment, kind: str, hours: int | None = None) -> tuple[str, str]:
    p = a.patient
    hi = f"Dear {p.first_name},"
    if kind == "confirmation":
        subject = f"Appointment confirmed: {a.scheduled_at:%d %b %Y, %I:%M %p}"
        intro = "Your appointment is confirmed. A calendar invite is attached."
    elif kind == "rescheduled":
        subject = f"Appointment changed: now {a.scheduled_at:%d %b %Y, %I:%M %p}"
        intro = "Your appointment has been moved to a new time. An updated calendar invite is attached."
    elif kind == "cancelled":
        subject = f"Appointment cancelled: {a.scheduled_at:%d %b %Y, %I:%M %p}"
        intro = "Your appointment has been cancelled. You can book a new one online at any time."
    else:  # reminder
        when = "tomorrow" if hours and hours >= 12 else "soon"
        subject = f"Reminder: appointment {when}, {a.scheduled_at:%d %b, %I:%M %p}"
        intro = (f"This is a reminder of your appointment {when}. "
                 "Please arrive 10 minutes early and bring any reports or medicines you are taking.")
    body = "\n".join([hi, "", intro, "", *_appt_lines(a), *_footer(a)])
    return subject, body


def staff_new_booking_message(a: Appointment) -> tuple[str, str]:
    p = a.patient
    subject = f"New online booking: {p.full_name}, {a.scheduled_at:%d %b %I:%M %p}"
    body = "\n".join([f"{p.full_name} ({p.mrn}) booked an appointment online.", "", *_appt_lines(a), "",
                      f"Open it: {base_url()}/appointments/{a.id}", "", "MediPurse"])
    return subject, body


def portal_link_message(p: Patient, token: str, invite: bool) -> tuple[str, str]:
    link = f"{base_url()}/portal/set-password?token={token}"
    if invite:
        subject = f"Your {p.clinic.name} patient account"
        intro = (f"{p.clinic.label} has set up an online account for you. With it you can book and "
                 "cancel appointments and get reminders by email.")
        action = "Create your password to get started"
    else:
        subject = "Reset your MediPurse password"
        intro = "We received a request to set or reset the password for your patient account."
        action = "Set a new password"
    body = "\n".join([f"Dear {p.first_name},", "", intro, "", f"{action} (link valid for 7 days):", link, "",
                      "If you did not expect this email, you can ignore it.", "", "MediPurse"])
    return subject, body


# ----------------------------------------------------------------------------
# queueing
# ----------------------------------------------------------------------------
_wake = threading.Event()


def queue(db, *, to: str, subject: str, body: str, kind: str,
          patient_id: int | None = None, appointment_id: int | None = None) -> Notification | None:
    if not to or "@" not in to:
        return None
    n = Notification(to_email=to.strip(), subject=subject, body=body, kind=kind,
                     patient_id=patient_id, appointment_id=appointment_id)
    db.add(n)
    db.commit()
    _wake.set()
    return n


def notify_patient(db, a: Appointment, kind: str, hours: int | None = None) -> Notification | None:
    p = a.patient
    if not p.email or not p.email_reminders:
        return None
    subject, body = appointment_message(a, kind, hours)
    return queue(db, to=p.email, subject=subject, body=body, kind=kind,
                 patient_id=p.id, appointment_id=a.id)


def notify_doctor_online_booking(db, a: Appointment) -> None:
    if a.doctor and a.doctor.email:
        subject, body = staff_new_booking_message(a)
        queue(db, to=a.doctor.email, subject=subject, body=body, kind="new-booking",
              patient_id=a.patient_id, appointment_id=a.id)


def mark_past_reminders(a: Appointment, now: datetime | None = None) -> None:
    """Reminders whose window has already started are covered by the confirmation email."""
    now = now or datetime.now()
    left = a.scheduled_at - now
    a.reminders_sent = ",".join(str(h) for h in reminder_hours() if left <= timedelta(hours=h))


# ----------------------------------------------------------------------------
# sending
# ----------------------------------------------------------------------------
def _ics(a: Appointment) -> bytes:
    start = a.scheduled_at
    end = start + timedelta(minutes=VISIT_MINUTES)
    title = f"Appointment at {a.patient.clinic.name}" + (f" with {a.doctor.display}" if a.doctor else "")

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MediPurse//EN", "METHOD:PUBLISH", "BEGIN:VEVENT",
             f"UID:appointment-{a.id}@medipurse", f"DTSTAMP:{datetime.utcnow():%Y%m%dT%H%M%SZ}",
             f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}",
             f"SUMMARY:{esc(title)}", f"LOCATION:{esc(_place(a))}",
             f"DESCRIPTION:{esc(a.reason or 'Clinic appointment')}",
             "BEGIN:VALARM", "TRIGGER:-PT2H", "ACTION:DISPLAY", "DESCRIPTION:Appointment reminder", "END:VALARM",
             "END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode()


def _html(body: str) -> str:
    paras = "".join(
        f'<p style="margin:0 0 12px">{"<br>".join(escape(x) for x in block.splitlines())}</p>'
        for block in body.split("\n\n"))
    return ('<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.5;color:#18181b;max-width:560px">'
            '<div style="font-weight:700;font-size:18px;color:#2563eb;margin-bottom:16px">MediPurse</div>'
            f"{paras}</div>")


def build_email(n: Notification) -> EmailMessage:
    msg = EmailMessage()
    sender = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or "no-reply@medipurse.local"
    msg["From"] = sender if "<" in sender else formataddr(("MediPurse", sender))
    msg["To"] = n.to_email
    msg["Subject"] = n.subject
    msg["Message-ID"] = make_msgid(domain="medipurse")
    msg.set_content(n.body)
    msg.add_alternative(_html(n.body), subtype="html")
    if n.kind in ("confirmation", "rescheduled") and n.appointment:
        msg.add_attachment(_ics(n.appointment), maintype="text", subtype="calendar",
                           filename="appointment.ics")
    return msg


def _deliver(msg: EmailMessage) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT") or 587)
    user, pw = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASSWORD", "")
    ctx = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=30, context=ctx)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        if os.environ.get("SMTP_TLS", "1") != "0":
            server.starttls(context=ctx)
    try:
        if user:
            server.login(user, pw)
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass


def send_one(db, n: Notification) -> None:
    n.attempts += 1
    try:
        msg = build_email(n)
        if smtp_configured():
            _deliver(msg)
            n.status = "sent"
        else:
            OUTBOX.mkdir(exist_ok=True)
            safe = "".join(ch if ch.isalnum() else "-" for ch in n.kind)
            (OUTBOX / f"{datetime.now():%Y%m%d-%H%M%S}-{n.id}-{safe}.eml").write_bytes(bytes(msg))
            n.status = "saved"
        n.sent_at, n.error = datetime.now(), ""
    except Exception as exc:  # network, auth, bad address...
        n.error = str(exc)[:500]
        n.status = "failed" if n.attempts >= MAX_ATTEMPTS else "queued"
    db.commit()


def process_queue(db) -> int:
    rows = db.query(Notification).filter(Notification.status == "queued").order_by(Notification.id).limit(50).all()
    for n in rows:
        send_one(db, n)
    return len(rows)


def send_due_reminders(db, now: datetime | None = None) -> int:
    now = now or datetime.now()
    hours = reminder_hours()
    rows = db.query(Appointment).filter(
        Appointment.status == "scheduled", Appointment.scheduled_at > now,
        Appointment.scheduled_at <= now + timedelta(hours=max(hours))).all()
    sent = 0
    for a in rows:
        done = set(filter(None, (a.reminders_sent or "").split(",")))
        due = [h for h in hours if str(h) not in done and a.scheduled_at - now <= timedelta(hours=h)]
        if not due:
            continue
        # one email even if several windows are due (e.g. app was off); mark them all
        notify_patient(db, a, "reminder", hours=min(due))
        a.reminders_sent = ",".join(sorted(done | {str(h) for h in due}, key=int, reverse=True))
        db.commit()
        sent += 1
    return sent


# ----------------------------------------------------------------------------
# background worker
# ----------------------------------------------------------------------------
_started = False


def _loop() -> None:
    last_reminder_check = 0.0
    while True:
        db = SessionLocal()
        try:
            if time.time() - last_reminder_check >= 60:
                send_due_reminders(db)
                last_reminder_check = time.time()
            process_queue(db)
        except Exception as exc:
            print(f"[notify] worker error: {exc}", flush=True)
        finally:
            db.close()
        _wake.wait(timeout=30)
        _wake.clear()


def start_worker() -> None:
    global _started
    if _started or os.environ.get("NOTIFY_WORKER", "1") == "0":
        return
    _started = True
    threading.Thread(target=_loop, name="notify-worker", daemon=True).start()


def outbox_path() -> Path:
    return OUTBOX
