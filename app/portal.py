"""Patient portal: patients sign up or accept an invite, then book, view and cancel
appointments and choose whether to get email reminders.

Patients have their own session key ("patient_id"), separate from staff sign-in.
"""

from __future__ import annotations

import secrets
from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request

from . import notify
from .db import Appointment, Clinic, Patient, User, check_password, hash_password
from .main import Redirect, _date, _int, _s, flash, get_db, redirect, templates

router = APIRouter(prefix="/portal")

# booking rules
OPEN, CLOSE = time(9, 0), time(17, 0)
BREAK = (time(13, 0), time(14, 0))
SLOT_MIN = 30
WORK_DAYS = {0, 1, 2, 3, 4, 5}          # Monday to Saturday
BOOK_AHEAD_DAYS = 30
MIN_LEAD = timedelta(hours=1)
CANCEL_CUTOFF = timedelta(hours=2)
MAX_UPCOMING = 3


def current_patient(request: Request, db=Depends(get_db)) -> Patient:
    pid = request.session.get("patient_id")
    p = db.get(Patient, pid) if pid else None
    if not p or not p.password_hash:
        request.session.pop("patient_id", None)
        raise Redirect("/portal/login")
    return p


def prender(request: Request, name: str, patient: Patient | None = None, **ctx):
    msgs = request.session.pop("flash", []) if "flash" in request.session else []
    return templates.TemplateResponse(request, name, {
        "patient": patient, "flashes": msgs, "path": request.url.path, "today": date.today(), **ctx})


def _login(request: Request, p: Patient) -> None:
    request.session.clear()
    request.session["patient_id"] = p.id


def _upcoming(p: Patient) -> list[Appointment]:
    now = datetime.now()
    return sorted([a for a in p.appointments if a.status == "scheduled" and a.scheduled_at >= now],
                  key=lambda a: a.scheduled_at)


# ----------------------------------------------------------------------------
# sign in / sign up / password
# ----------------------------------------------------------------------------
@router.get("/login")
def login_page(request: Request):
    return prender(request, "portal/login.html")


@router.post("/login")
async def login(request: Request, db=Depends(get_db)):
    form = await request.form()
    email, pw = _s(form, "email").lower(), _s(form, "password")
    for p in db.query(Patient).filter(Patient.email.ilike(email), Patient.password_hash != "").all():
        if check_password(pw, p.password_hash):
            _login(request, p)
            return redirect("/portal")
    flash(request, "Incorrect email or password.", "error")
    return redirect("/portal/login")


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return redirect("/portal/login")


@router.get("/register")
def register_page(request: Request, db=Depends(get_db)):
    return prender(request, "portal/register.html", clinics=db.query(Clinic).order_by(Clinic.name).all())


@router.post("/register")
async def register(request: Request, db=Depends(get_db)):
    form = await request.form()
    email = _s(form, "email").lower()
    clinic = db.get(Clinic, _int(form.get("clinic_id")) or 0)
    err = None
    if not _s(form, "first_name") or not email or "@" not in email or not _s(form, "phone"):
        err = "First name, email and phone are required."
    elif not clinic:
        err = "Choose your clinic."
    elif len(_s(form, "password")) < 8:
        err = "Password must be at least 8 characters."
    elif _s(form, "password") != _s(form, "password2"):
        err = "The two passwords do not match."
    if err:
        flash(request, err, "error")
        return redirect("/portal/register")

    existing = db.query(Patient).filter(Patient.email.ilike(email), Patient.clinic_id == clinic.id).first()
    if existing:
        # Already a patient (e.g. registered at the desk): prove the email by sending a link.
        _send_password_link(db, existing, invite=False)
        flash(request, "You are already registered with this clinic. We emailed you a link to set your "
                       "password and finish signing up.")
        return redirect("/portal/login")

    p = Patient(clinic_id=clinic.id, first_name=_s(form, "first_name"), last_name=_s(form, "last_name"),
                dob=_date(form.get("dob")), sex=_s(form, "sex"), phone=_s(form, "phone"), email=email,
                password_hash=hash_password(_s(form, "password")), email_reminders=1)
    db.add(p)
    db.commit()
    _login(request, p)
    flash(request, f"Welcome, {p.first_name}. Your patient ID is {p.mrn}. You can book an appointment now.")
    return redirect("/portal")


def _send_password_link(db, p: Patient, invite: bool) -> None:
    token = secrets.token_urlsafe(32)
    p.portal_token, p.token_expires = token, datetime.now() + timedelta(days=7)
    db.commit()
    subject, body = notify.portal_link_message(p, token, invite=invite)
    notify.queue(db, to=p.email, subject=subject, body=body, kind="invite" if invite else "password",
                 patient_id=p.id)


@router.get("/forgot")
def forgot_page(request: Request):
    return prender(request, "portal/forgot.html")


@router.post("/forgot")
async def forgot(request: Request, db=Depends(get_db)):
    form = await request.form()
    email = _s(form, "email").lower()
    for p in db.query(Patient).filter(Patient.email.ilike(email)).limit(5).all():
        _send_password_link(db, p, invite=False)
    # same answer whether or not the email exists
    flash(request, "If that email is registered, a link to set your password is on its way.")
    return redirect("/portal/login")


def _token_patient(db, token: str) -> Patient | None:
    if not token or len(token) < 20:
        return None
    p = db.query(Patient).filter(Patient.portal_token == token).first()
    if not p or not p.token_expires or p.token_expires < datetime.now():
        return None
    return p


@router.get("/set-password")
def set_password_page(request: Request, token: str = "", db=Depends(get_db)):
    p = _token_patient(db, token)
    if not p:
        flash(request, "This link has expired or was already used. Request a new one below.", "error")
        return redirect("/portal/forgot")
    return prender(request, "portal/set_password.html", token=token, who=p)


@router.post("/set-password")
async def set_password(request: Request, db=Depends(get_db)):
    form = await request.form()
    token = _s(form, "token")
    p = _token_patient(db, token)
    if not p:
        flash(request, "This link has expired or was already used. Request a new one below.", "error")
        return redirect("/portal/forgot")
    pw = _s(form, "password")
    if len(pw) < 8 or pw != _s(form, "password2"):
        flash(request, "Use at least 8 characters and type the same password twice.", "error")
        return redirect(f"/portal/set-password?token={token}")
    p.password_hash, p.portal_token, p.token_expires = hash_password(pw), "", None
    db.commit()
    _login(request, p)
    flash(request, "Your password is set. Welcome to your patient account.")
    return redirect("/portal")


# ----------------------------------------------------------------------------
# home
# ----------------------------------------------------------------------------
@router.get("")
def home(request: Request, p: Patient = Depends(current_patient)):
    upcoming = _upcoming(p)
    past = sorted([a for a in p.appointments if a not in upcoming], key=lambda a: a.scheduled_at, reverse=True)
    soon = [a for a in upcoming if a.scheduled_at - datetime.now() <= timedelta(hours=48)]
    rx = sorted([r for r in p.prescriptions if r.status == "active"], key=lambda r: r.prescribed_at, reverse=True)
    return prender(request, "portal/home.html", p, upcoming=upcoming, past=past[:20], soon=soon,
                   rx=rx, cutoff=CANCEL_CUTOFF, now=datetime.now())


# ----------------------------------------------------------------------------
# booking
# ----------------------------------------------------------------------------
def _clinic_doctors(db, p: Patient) -> list[User]:
    return db.query(User).filter(User.role == "doctor", User.active == 1,
                                 User.clinic_id == p.clinic_id).order_by(User.name).all()


def _slots(db, doctor_id: int, day: date, patient: Patient) -> list[datetime]:
    if day.weekday() not in WORK_DAYS:
        return []
    start, end = datetime.combine(day, OPEN), datetime.combine(day, CLOSE)
    taken = [a.scheduled_at for a in db.query(Appointment).filter(
        Appointment.status == "scheduled",
        Appointment.scheduled_at > start - timedelta(minutes=SLOT_MIN),
        Appointment.scheduled_at < end + timedelta(minutes=SLOT_MIN),
        (Appointment.doctor_id == doctor_id) | (Appointment.patient_id == patient.id)).all()]
    earliest = datetime.now() + MIN_LEAD
    out, t = [], start
    while t + timedelta(minutes=SLOT_MIN) <= end:
        in_break = BREAK[0] <= t.time() < BREAK[1]
        clash = any(abs((t - x).total_seconds()) < SLOT_MIN * 60 for x in taken)
        if not in_break and not clash and t >= earliest:
            out.append(t)
        t += timedelta(minutes=SLOT_MIN)
    return out


def _days() -> list[date]:
    t = date.today()
    return [t + timedelta(days=i) for i in range(BOOK_AHEAD_DAYS + 1) if (t + timedelta(days=i)).weekday() in WORK_DAYS]


@router.get("/book")
def book_page(request: Request, doctor_id: int | None = None, day: str = "", db=Depends(get_db),
              p: Patient = Depends(current_patient)):
    doctors = _clinic_doctors(db, p)
    doc = next((d for d in doctors if d.id == doctor_id), doctors[0] if len(doctors) == 1 else None)
    days = _days()
    d = _date(day)
    if d not in days:
        d = None
    if doc and not d:
        d = next((x for x in days if _slots(db, doc.id, x, p)), None)
    slots = _slots(db, doc.id, d, p) if doc and d else []
    return prender(request, "portal/book.html", p, doctors=doctors, doc=doc, days=days, day=d, slots=slots,
                   n_upcoming=len(_upcoming(p)), max_upcoming=MAX_UPCOMING)


@router.post("/book")
async def book(request: Request, db=Depends(get_db), p: Patient = Depends(current_patient)):
    form = await request.form()
    doc_id = _int(form.get("doctor_id"))
    doc = next((d for d in _clinic_doctors(db, p) if d.id == doc_id), None)
    try:
        when = datetime.fromisoformat(_s(form, "slot"))
    except ValueError:
        when = None
    back = f"/portal/book?doctor_id={doc_id or ''}"
    if not doc or not when:
        flash(request, "Choose a doctor and a time.", "error")
        return redirect(back)
    if len(_upcoming(p)) >= MAX_UPCOMING:
        flash(request, f"You can have up to {MAX_UPCOMING} upcoming appointments. Cancel one to book another.", "error")
        return redirect("/portal")
    if when not in _slots(db, doc.id, when.date(), p):
        flash(request, "Sorry, that time is no longer available. Please pick another.", "error")
        return redirect(f"{back}&day={when.date().isoformat()}")
    a = Appointment(patient_id=p.id, doctor_id=doc.id, scheduled_at=when, reason=_s(form, "reason")[:300],
                    source="online", acknowledged=0)
    notify.mark_past_reminders(a)
    db.add(a)
    db.commit()
    sent = notify.notify_patient(db, a, "confirmation")
    notify.notify_doctor_online_booking(db, a)
    flash(request, f"Booked with {doc.display} on {when:%A %d %B at %I:%M %p}."
                   + (f" A confirmation has been emailed to {p.email}." if sent else ""))
    return redirect("/portal")


@router.post("/appointments/{aid}/cancel")
def cancel(aid: int, request: Request, db=Depends(get_db), p: Patient = Depends(current_patient)):
    a = db.get(Appointment, aid)
    if not a or a.patient_id != p.id:
        raise HTTPException(404)
    if a.status != "scheduled" or a.scheduled_at - datetime.now() < CANCEL_CUTOFF:
        flash(request, "This appointment can no longer be cancelled online. Please call the clinic.", "error")
        return redirect("/portal")
    a.status, a.acknowledged = "cancelled", 1
    a.notes = (a.notes + "\n" if a.notes else "") + f"Cancelled by patient online on {datetime.now():%d %b %Y %I:%M %p}."
    db.commit()
    notify.notify_patient(db, a, "cancelled")
    flash(request, f"Your appointment on {a.scheduled_at:%d %B at %I:%M %p} is cancelled.")
    return redirect("/portal")


# ----------------------------------------------------------------------------
# profile
# ----------------------------------------------------------------------------
@router.get("/profile")
def profile_page(request: Request, p: Patient = Depends(current_patient)):
    return prender(request, "portal/profile.html", p)


@router.post("/profile")
async def profile(request: Request, db=Depends(get_db), p: Patient = Depends(current_patient)):
    form = await request.form()
    if _s(form, "email") and "@" in _s(form, "email"):
        p.email = _s(form, "email").lower()
    for f in ("phone", "address", "emergency_contact"):
        setattr(p, f, _s(form, f))
    p.email_reminders = 1 if form.get("email_reminders") else 0
    new = _s(form, "new_password")
    if new:
        if not check_password(_s(form, "current_password"), p.password_hash):
            flash(request, "Your current password is not correct.", "error")
            return redirect("/portal/profile")
        if len(new) < 8:
            flash(request, "New password must be at least 8 characters.", "error")
            return redirect("/portal/profile")
        p.password_hash = hash_password(new)
    db.commit()
    flash(request, "Your details are saved.")
    return redirect("/portal/profile")
