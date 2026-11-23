"""Email notifications.

Patients: appointment confirmations, changes, cancellations and reminders; new and
stopped prescriptions; a feedback request after a diagnosis or a completed visit;
portal invites and password links.
Staff: new online bookings, urgent symptoms reported in a patient check-in, and low
feedback ratings.

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

import re
import secrets

from .db import ROOT, Appointment, FeedbackRequest, Notification, Patient, SessionLocal, User, role_clause

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


_URL = re.compile(r"(https?://[^\s<]+)")


def _line_html(x: str) -> str:
    return _URL.sub(r'<a href="\1" style="color:#2563eb">\1</a>', escape(x))


def _html(body: str) -> str:
    paras = "".join(
        f'<p style="margin:0 0 12px">{"<br>".join(_line_html(x) for x in block.splitlines())}</p>'
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


# ----------------------------------------------------------------------------
# prescriptions, feedback and staff alerts
# ----------------------------------------------------------------------------
FEEDBACK_DAYS = 14          # how long a feedback link works
FEEDBACK_GAP = timedelta(hours=12)   # at most one feedback request per patient per visit/half-day
LOW_RATING = 2              # ratings at or below this alert the doctor


def _clinic_footer(p: Patient) -> list[str]:
    out = ["", f"Your patient account: {base_url()}/portal"]
    if p.clinic.phone:
        out.append(f"Questions? Call {p.clinic.label}: {p.clinic.phone}")
    out += ["", "MediPurse"]
    return out


def _wants_email(p: Patient) -> bool:
    return bool(p.email and "@" in p.email and p.email_reminders)


def _rx_line(r) -> str:
    bits = [b for b in (r.dosage, r.frequency, f"for {r.duration}" if r.duration else "") if b]
    line = f"- {r.medicine}" + (f": {', '.join(bits)}" if bits else "")
    if getattr(r, "quantity", 0):
        line += f" ({r.quantity} in total)"
    if r.instructions:
        line += f"\n  {r.instructions}"
    supply = getattr(r, "supply", "")
    if getattr(r, "dispense_status", "") == "pending":
        from .pharmacy import availability
        av = availability(r)
        line += {"full": "\n  Collect it from the clinic pharmacy.",
                 "some": "\n  The clinic pharmacy has some of this; you may need to buy the rest from another pharmacy.",
                 "none": "\n  Not available at the clinic. Please buy it from another pharmacy."}[av]
    elif supply == "clinic":
        line += "\n  Given to you by the clinic pharmacy."
    elif supply == "partial":
        line += f"\n  The clinic gave you {r.dispensed}. Please buy the remaining {r.quantity - r.dispensed} from another pharmacy."
    elif supply == "outside":
        line += "\n  Not available at the clinic. Please buy it from another pharmacy."
    return line


def prescription_message(p: Patient, rxs: list, doctor: User | None) -> tuple[str, str]:
    names = ", ".join(r.medicine for r in rxs)
    subject = f"New prescription: {names}" if len(names) <= 60 else f"New prescription: {len(rxs)} medicines"
    by = f" by {doctor.display}" if doctor and doctor.has("doctor") else ""
    when = rxs[0].prescribed_at or datetime.now()
    body = "\n".join([f"Dear {p.first_name},", "",
                      f"The following medicine{'s were' if len(rxs) != 1 else ' was'} prescribed for you{by} "
                      f"at {p.clinic.label} on {when:%d %B %Y, %I:%M %p}.", "",
                      *[_rx_line(r) for r in rxs], "",
                      "Take your medicines exactly as prescribed. Do not stop or change them without asking "
                      "your doctor. If you notice side effects such as a rash, swelling or breathing trouble, "
                      "contact the clinic straight away (in an emergency call 112).",
                      *_clinic_footer(p)])
    return subject, body


def prescription_stopped_message(p: Patient, r) -> tuple[str, str]:
    subject = f"Medicine stopped: {r.medicine}"
    body = "\n".join([f"Dear {p.first_name},", "",
                      f"Your doctor has stopped {r.medicine} as of {(r.stopped_on or datetime.now()):%d %B %Y}.",
                      "Please stop taking it unless your doctor tells you otherwise. Your other medicines "
                      "are not affected.", *_clinic_footer(p)])
    return subject, body


def notify_prescriptions(db, p: Patient, rxs: list, doctor: User | None) -> Notification | None:
    if not rxs or not _wants_email(p):
        return None
    subject, body = prescription_message(p, rxs, doctor)
    return queue(db, to=p.email, subject=subject, body=body, kind="prescription", patient_id=p.id,
                 appointment_id=rxs[0].appointment_id)


def notify_prescription_stopped(db, p: Patient, r) -> Notification | None:
    if not _wants_email(p):
        return None
    subject, body = prescription_stopped_message(p, r)
    return queue(db, to=p.email, subject=subject, body=body, kind="prescription-stopped", patient_id=p.id,
                 appointment_id=r.appointment_id)


def feedback_message(p: Patient, fr: FeedbackRequest, reason: str) -> tuple[str, str]:
    link = f"{base_url()}/portal/feedback/{fr.token}"
    who = f" with {fr.doctor.display}" if fr.doctor else ""
    stars = [f"{'★' * n}{'☆' * (5 - n)}  {label}: {link}?rating={n}"
             for n, label in [(5, "Excellent"), (4, "Good"), (3, "Okay"), (2, "Poor"), (1, "Very poor")]]
    body = "\n".join([f"Dear {p.first_name},", "",
                      f"Thank you for your visit to {p.clinic.label}{who}. {reason}", "",
                      "How was your care? It takes 30 seconds and helps us improve. Tap a rating:", "",
                      *stars, "",
                      f"This link works for {FEEDBACK_DAYS} days and can be used once.",
                      *_clinic_footer(p)])
    return f"How was your visit to {p.clinic.name}?", body


def request_feedback(db, p: Patient, *, doctor: User | None = None, appointment_id: int | None = None,
                     diagnosis_id: int | None = None, reason: str = "") -> Notification | None:
    """Email a one-time feedback link. Skipped if the patient was asked recently or about
    this appointment already, so several diagnoses in one visit send one email."""
    if not _wants_email(p):
        return None
    recent = db.query(FeedbackRequest).filter(FeedbackRequest.patient_id == p.id)
    if appointment_id and recent.filter(FeedbackRequest.appointment_id == appointment_id).first():
        return None
    if recent.filter(FeedbackRequest.created_at >= datetime.now() - FEEDBACK_GAP).first():
        return None
    fr = FeedbackRequest(patient_id=p.id, appointment_id=appointment_id, diagnosis_id=diagnosis_id,
                         doctor_id=doctor.id if doctor and doctor.has("doctor") else None,
                         token=secrets.token_urlsafe(32), expires_at=datetime.now() + timedelta(days=FEEDBACK_DAYS))
    db.add(fr)
    db.commit()
    subject, body = feedback_message(p, fr, reason)
    return queue(db, to=p.email, subject=subject, body=body, kind="feedback-request", patient_id=p.id,
                 appointment_id=appointment_id)


def _clinic_doctors(db, p: Patient) -> list[User]:
    return db.query(User).filter(role_clause("doctor"), User.active == 1, User.clinic_id == p.clinic_id).all()


def alert_low_feedback(db, fb, doctor: User | None) -> int:
    """A rating of 1-2 from a patient emails their doctor (or the clinic's doctors)."""
    if fb.rating > LOW_RATING:
        return 0
    p = fb.patient
    to = [doctor] if doctor and doctor.email else _clinic_doctors(db, p)
    subject = f"Low feedback ({fb.rating}/5) from {p.full_name}"
    body = "\n".join([f"{p.full_name} ({p.mrn}) rated their visit {fb.rating} out of 5.", "",
                      f"Comments: {fb.comments or '(none)'}", "",
                      f"Open the patient: {base_url()}/patients/{p.id}?tab=feedback", "", "MediPurse"])
    n = 0
    for u in to:
        if u.email:
            queue(db, to=u.email, subject=subject, body=body, kind="feedback-alert", patient_id=p.id)
            n += 1
    return n


def alert_red_flags(db, intake) -> int:
    """Urgent symptoms in a patient's check-in email the appointment doctor (or clinic doctors)."""
    if not intake.red_flags:
        return 0
    p = intake.patient
    doc = intake.appointment.doctor if intake.appointment and intake.appointment.doctor else None
    to = [doc] if doc and doc.email else _clinic_doctors(db, p)
    subject = f"Urgent symptoms reported by {p.full_name}"
    body = "\n".join([f"{p.full_name} ({p.mrn}) reported possible red-flag symptoms in their symptom check-in:", "",
                      intake.red_flags, "",
                      "The patient was told to call 112 or go to the emergency department if it is happening now.",
                      "Please review and contact the patient if needed.", "",
                      f"Read the history: {base_url()}/history/{intake.id}", "", "MediPurse"])
    n = 0
    for u in to:
        if u.email:
            queue(db, to=u.email, subject=subject, body=body, kind="urgent-alert", patient_id=p.id,
                  appointment_id=intake.appointment_id)
            n += 1
    return n


def dispensed_message(p: Patient, rxs: list) -> tuple[str, str]:
    given = [r for r in rxs if r.dispensed]
    outside = [r for r in rxs if r.supply in ("outside", "partial")]
    lines = [f"Dear {p.first_name},", ""]
    if given:
        lines += ["The clinic pharmacy has given you:", *[f"- {r.medicine}: {r.dispensed}" for r in given], ""]
    if outside:
        lines += ["Please buy these from another pharmacy:",
                  *[f"- {r.medicine}" + (f": {r.quantity - r.dispensed} more" if r.supply == "partial" else
                                         (f": {r.quantity}" if r.quantity else "")) for r in outside], ""]
    lines += ["Take your medicines as your doctor prescribed. Ask the pharmacist if anything is unclear.",
              *_clinic_footer(p)]
    subject = "Your medicines from the clinic pharmacy" if given else "Medicines to buy from another pharmacy"
    return subject, "\n".join(lines)


def notify_dispensed(db, p: Patient, rxs: list) -> Notification | None:
    if not rxs or not _wants_email(p):
        return None
    subject, body = dispensed_message(p, rxs)
    return queue(db, to=p.email, subject=subject, body=body, kind="dispensed", patient_id=p.id,
                 appointment_id=rxs[0].appointment_id)
