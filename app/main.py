"""MediPurse - simple hospital management system with AI diagnosis support.

Run from the repository root:
    .venv\\Scripts\\python -m uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()

from . import notify  # noqa: E402
from .ai import AI  # noqa: E402
from .db import (APPT_STATUSES, ROLES, AIAnalysis, Appointment, Clinic,  # noqa: E402
                 Diagnosis, Feedback, Hospital, Notification, Patient, Prescription,
                 SessionLocal, User, check_password, hash_password, init_db)

app = FastAPI(title="MediPurse")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
                   max_age=60 * 60 * 12)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _fmt_dt(v, fmt="%d %b %Y, %I:%M %p"):
    return v.strftime(fmt) if v else ""


def _fmt_d(v):
    return v.strftime("%d %b %Y") if v else ""


templates.env.filters["dt"] = _fmt_dt
templates.env.filters["d"] = _fmt_d
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}


@app.on_event("startup")
def _startup() -> None:
    init_db()
    from .seed import seed_if_empty, seed_portal_demo, seed_prescriptions_if_empty
    seed_if_empty()
    seed_prescriptions_if_empty()
    seed_portal_demo()
    AI.load_async()
    notify.start_worker()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Redirect(Exception):
    def __init__(self, url: str):
        self.url = url


@app.exception_handler(Redirect)
def _redirect_handler(request: Request, exc: Redirect):
    return RedirectResponse(exc.url, status_code=303)


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def flash(request: Request, msg: str, kind: str = "ok") -> None:
    request.session.setdefault("flash", []).append([kind, msg])


def current_user(request: Request, db=Depends(get_db)) -> User:
    uid = request.session.get("uid")
    user = db.get(User, uid) if uid else None
    if not user or not user.active:
        request.session.pop("uid", None)
        raise Redirect("/login")
    return user


def require(user: User, *roles: str) -> None:
    if user.role not in roles:
        raise HTTPException(403, "You do not have access to this page.")


def online_bookings_query(db, user: User):
    q = scoped_appointments(db, user).filter(Appointment.source == "online", Appointment.acknowledged == 0,
                                             Appointment.status == "scheduled")
    if user.role == "doctor":
        q = q.filter(Appointment.doctor_id == user.id)
    return q


def render(request: Request, name: str, user: User | None = None, **ctx):
    msgs = request.session.pop("flash", []) if "flash" in request.session else []
    if user is not None and "new_online" not in ctx:
        db = SessionLocal()
        try:
            ctx["new_online"] = online_bookings_query(db, user).count()
        finally:
            db.close()
    return templates.TemplateResponse(request, name, {
        "user": user, "flashes": msgs, "ROLES": ROLES, "path": request.url.path,
        "ai_status": AI.status, "llm_on": AI.llm_configured, "today": date.today(), **ctx})


def scoped_patients(db, user: User):
    q = db.query(Patient)
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    return q


def get_patient(db, user: User, pid: int) -> Patient:
    p = db.get(Patient, pid)
    if not p or (user.role != "admin" and p.clinic_id != user.clinic_id):
        raise HTTPException(404, "Patient not found")
    return p


def scoped_appointments(db, user: User):
    q = db.query(Appointment).join(Patient)
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    return q


def doctors_for(db, user: User, clinic_id: int | None = None):
    q = db.query(User).filter(User.role == "doctor", User.active == 1)
    cid = clinic_id or (user.clinic_id if user.role != "admin" else None)
    if cid:
        q = q.filter(User.clinic_id == cid)
    return q.order_by(User.name).all()


def visible_clinics(db, user: User):
    q = db.query(Clinic)
    if user.role != "admin":
        q = q.filter(Clinic.id == user.clinic_id)
    return q.order_by(Clinic.name).all()


def _int(v) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _date(v) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


def _s(form, key) -> str:
    return str(form.get(key, "") or "").strip()


# ----------------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------------
@app.get("/login")
def login_page(request: Request):
    return render(request, "login.html")


@app.post("/login")
async def login(request: Request, db=Depends(get_db)):
    form = await request.form()
    email = _s(form, "email").lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.active or not check_password(_s(form, "password"), user.password_hash):
        flash(request, "Incorrect email or password.", "error")
        return redirect("/login")
    request.session.clear()
    request.session["uid"] = user.id
    return redirect("/")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return redirect("/login")


# ----------------------------------------------------------------------------
# dashboard
# ----------------------------------------------------------------------------
@app.get("/")
def dashboard(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    start = datetime.combine(date.today(), datetime.min.time())
    end = start + timedelta(days=1)
    appts = scoped_appointments(db, user).filter(
        Appointment.scheduled_at >= start, Appointment.scheduled_at < end)
    if user.role == "doctor":
        appts = appts.filter(Appointment.doctor_id == user.id)
    appts = appts.order_by(Appointment.scheduled_at).all()
    upcoming = scoped_appointments(db, user).filter(
        Appointment.scheduled_at >= end, Appointment.status == "scheduled")
    if user.role == "doctor":
        upcoming = upcoming.filter(Appointment.doctor_id == user.id)
    stats = {
        "patients": scoped_patients(db, user).count(),
        "today": len(appts),
        "upcoming": upcoming.count(),
        "active_dx": db.query(Diagnosis).join(Patient).filter(
            Diagnosis.status == "active",
            *( [] if user.role == "admin" else [Patient.clinic_id == user.clinic_id])).count(),
    }
    rxq = db.query(Prescription).join(Patient).filter(Prescription.prescribed_at >= start,
                                                      Prescription.prescribed_at < end)
    if user.role != "admin":
        rxq = rxq.filter(Patient.clinic_id == user.clinic_id)
    stats["rx_today"] = rxq.count()
    if user.role == "admin":
        stats["hospitals"] = db.query(Hospital).count()
        stats["clinics"] = db.query(Clinic).count()
        stats["staff"] = db.query(User).count()
    recent_patients = scoped_patients(db, user).order_by(Patient.created_at.desc()).limit(5).all()
    online = online_bookings_query(db, user).order_by(Appointment.scheduled_at).all()
    return render(request, "dashboard.html", user, appts=appts, stats=stats,
                  recent_patients=recent_patients, online=online)


# ----------------------------------------------------------------------------
# hospitals (registrar)
# ----------------------------------------------------------------------------
HOSPITAL_FIELDS = ["name", "registration_no", "address", "city", "phone", "email"]


@app.get("/hospitals")
def hospitals(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    rows = db.query(Hospital).order_by(Hospital.name).all()
    return render(request, "hospitals.html", user, hospitals=rows)


@app.get("/hospitals/new")
def hospital_new(request: Request, user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "hospital_form.html", user, h=None)


@app.post("/hospitals/new")
async def hospital_create(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    form = await request.form()
    if not _s(form, "name"):
        flash(request, "Hospital name is required.", "error")
        return redirect("/hospitals/new")
    h = Hospital(**{f: _s(form, f) for f in HOSPITAL_FIELDS})
    db.add(h)
    db.commit()
    flash(request, f"Hospital “{h.name}” registered. Now add its clinics.")
    return redirect(f"/hospitals/{h.id}")


@app.get("/hospitals/{hid}")
def hospital_detail(hid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    h = db.get(Hospital, hid)
    if not h:
        raise HTTPException(404, "Hospital not found")
    return render(request, "hospital_detail.html", user, h=h)


@app.get("/hospitals/{hid}/edit")
def hospital_edit(hid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "hospital_form.html", user, h=db.get(Hospital, hid))


@app.post("/hospitals/{hid}/edit")
async def hospital_update(hid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    h = db.get(Hospital, hid)
    form = await request.form()
    for f in HOSPITAL_FIELDS:
        setattr(h, f, _s(form, f))
    db.commit()
    flash(request, "Hospital details saved.")
    return redirect(f"/hospitals/{hid}")


# ----------------------------------------------------------------------------
# clinics
# ----------------------------------------------------------------------------
CLINIC_FIELDS = ["name", "specialty", "address", "phone"]


@app.get("/clinics")
def clinics(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    rows = db.query(Clinic).order_by(Clinic.name).all()
    return render(request, "clinics.html", user, clinics=rows)


@app.get("/clinics/new")
def clinic_new(request: Request, hospital_id: int | None = None, db=Depends(get_db),
               user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "clinic_form.html", user, c=None, hospital_id=hospital_id,
                  hospitals=db.query(Hospital).order_by(Hospital.name).all())


@app.post("/clinics/new")
async def clinic_create(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    form = await request.form()
    if not _s(form, "name"):
        flash(request, "Clinic name is required.", "error")
        return redirect("/clinics/new")
    c = Clinic(hospital_id=_int(form.get("hospital_id")), **{f: _s(form, f) for f in CLINIC_FIELDS})
    db.add(c)
    db.commit()
    flash(request, f"Clinic “{c.name}” registered.")
    return redirect(f"/clinics/{c.id}")


@app.get("/clinics/{cid}")
def clinic_detail(cid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    if user.role != "admin" and user.clinic_id != cid:
        raise HTTPException(403)
    c = db.get(Clinic, cid)
    if not c:
        raise HTTPException(404)
    n_patients = db.query(Patient).filter(Patient.clinic_id == cid).count()
    return render(request, "clinic_detail.html", user, c=c, n_patients=n_patients)


@app.get("/clinics/{cid}/edit")
def clinic_edit(cid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "clinic_form.html", user, c=db.get(Clinic, cid), hospital_id=None,
                  hospitals=db.query(Hospital).order_by(Hospital.name).all())


@app.post("/clinics/{cid}/edit")
async def clinic_update(cid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    c = db.get(Clinic, cid)
    form = await request.form()
    for f in CLINIC_FIELDS:
        setattr(c, f, _s(form, f))
    c.hospital_id = _int(form.get("hospital_id"))
    db.commit()
    flash(request, "Clinic details saved.")
    return redirect(f"/clinics/{cid}")


# ----------------------------------------------------------------------------
# staff
# ----------------------------------------------------------------------------
@app.get("/staff")
def staff(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    rows = db.query(User).order_by(User.role, User.name).all()
    return render(request, "staff.html", user, staff=rows)


@app.get("/staff/new")
def staff_new(request: Request, clinic_id: int | None = None, db=Depends(get_db),
              user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "staff_form.html", user, s=None, clinic_id=clinic_id,
                  clinics=db.query(Clinic).order_by(Clinic.name).all())


@app.post("/staff/new")
async def staff_create(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    form = await request.form()
    email = _s(form, "email").lower()
    role = _s(form, "role")
    err = None
    if not _s(form, "name") or not email:
        err = "Name and email are required."
    elif role not in ROLES:
        err = "Choose a role."
    elif len(_s(form, "password")) < 6:
        err = "Password must be at least 6 characters."
    elif db.query(User).filter(User.email == email).first():
        err = "An account with this email already exists."
    elif role != "admin" and not _int(form.get("clinic_id")):
        err = "Doctors and receptionists must belong to a clinic."
    if err:
        flash(request, err, "error")
        return redirect("/staff/new")
    s = User(name=_s(form, "name"), email=email, role=role, specialty=_s(form, "specialty"),
             clinic_id=_int(form.get("clinic_id")) if role != "admin" else None,
             password_hash=hash_password(_s(form, "password")))
    db.add(s)
    db.commit()
    flash(request, f"Account created for {s.name}.")
    return redirect("/staff")


@app.get("/staff/{sid}/edit")
def staff_edit(sid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    return render(request, "staff_form.html", user, s=db.get(User, sid), clinic_id=None,
                  clinics=db.query(Clinic).order_by(Clinic.name).all())


@app.post("/staff/{sid}/edit")
async def staff_update(sid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "admin")
    s = db.get(User, sid)
    form = await request.form()
    s.name = _s(form, "name") or s.name
    s.specialty = _s(form, "specialty")
    if _s(form, "role") in ROLES and s.id != user.id:
        s.role = _s(form, "role")
    s.clinic_id = _int(form.get("clinic_id")) if s.role != "admin" else None
    if _s(form, "password"):
        s.password_hash = hash_password(_s(form, "password"))
    if s.id != user.id:
        s.active = 1 if form.get("active") else 0
    db.commit()
    flash(request, "Staff account saved.")
    return redirect("/staff")


# ----------------------------------------------------------------------------
# patients
# ----------------------------------------------------------------------------
PATIENT_FIELDS = ["first_name", "last_name", "sex", "phone", "email", "address",
                  "blood_group", "allergies", "emergency_contact"]


@app.get("/patients")
def patients(request: Request, q: str = "", db=Depends(get_db), user: User = Depends(current_user)):
    rows = scoped_patients(db, user)
    if q:
        like = f"%{q.strip()}%"
        num = q.strip().upper().removeprefix("P-").lstrip("0")
        conds = [Patient.first_name.ilike(like), Patient.last_name.ilike(like), Patient.phone.ilike(like)]
        if num.isdigit():
            conds.append(Patient.id == int(num))
        rows = rows.filter(or_(*conds))
    rows = rows.order_by(Patient.last_name, Patient.first_name).all()
    return render(request, "patients.html", user, patients=rows, q=q)


@app.get("/patients/new")
def patient_new(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    return render(request, "patient_form.html", user, p=None, clinics=visible_clinics(db, user))


@app.post("/patients/new")
async def patient_create(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    form = await request.form()
    clinic_id = _int(form.get("clinic_id")) if user.role == "admin" else user.clinic_id
    if not _s(form, "first_name") or not clinic_id:
        flash(request, "First name and clinic are required.", "error")
        return redirect("/patients/new")
    p = Patient(clinic_id=clinic_id, dob=_date(form.get("dob")),
                email_reminders=1 if form.get("email_reminders") else 0,
                **{f: _s(form, f) for f in PATIENT_FIELDS})
    db.add(p)
    db.commit()
    flash(request, f"Patient {p.full_name} registered ({p.mrn}).")
    return redirect(f"/patients/{p.id}")


@app.get("/patients/{pid}")
def patient_detail(pid: int, request: Request, tab: str = "overview", db=Depends(get_db),
                   user: User = Depends(current_user)):
    p = get_patient(db, user, pid)
    now = datetime.now()
    appts = sorted(p.appointments, key=lambda a: a.scheduled_at, reverse=True)
    upcoming = sorted([a for a in appts if a.scheduled_at >= now and a.status == "scheduled"],
                      key=lambda a: a.scheduled_at)
    past = [a for a in appts if a not in upcoming]
    dx = sorted(p.diagnoses, key=lambda d: d.diagnosed_on, reverse=True)
    rx = sorted(p.prescriptions, key=lambda r: r.prescribed_at, reverse=True)
    return render(request, "patient_detail.html", user, p=p, tab=tab,
                  active_rx=[r for r in rx if r.status == "active"],
                  past_rx=[r for r in rx if r.status != "active"],
                  upcoming=upcoming, past=past,
                  active_dx=[d for d in dx if d.status == "active"],
                  past_dx=[d for d in dx if d.status != "active"],
                  analyses=sorted(p.analyses, key=lambda a: a.created_at, reverse=True),
                  feedback=sorted(p.feedback, key=lambda f: f.created_at, reverse=True))


@app.post("/patients/{pid}/invite")
def patient_invite(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    p = get_patient(db, user, pid)
    if not p.email:
        flash(request, "Add the patient's email address first.", "error")
        return redirect(f"/patients/{pid}/edit")
    token = secrets.token_urlsafe(32)
    p.portal_token, p.token_expires = token, datetime.now() + timedelta(days=7)
    db.commit()
    subject, body = notify.portal_link_message(p, token, invite=True)
    notify.queue(db, to=p.email, subject=subject, body=body, kind="invite", patient_id=p.id)
    flash(request, f"Patient portal invitation emailed to {p.email}.")
    return redirect(f"/patients/{pid}")


@app.get("/patients/{pid}/edit")
def patient_edit(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    return render(request, "patient_form.html", user, p=get_patient(db, user, pid),
                  clinics=visible_clinics(db, user))


@app.post("/patients/{pid}/edit")
async def patient_update(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    p = get_patient(db, user, pid)
    form = await request.form()
    for f in PATIENT_FIELDS:
        setattr(p, f, _s(form, f))
    p.email_reminders = 1 if form.get("email_reminders") else 0
    p.dob = _date(form.get("dob"))
    if user.role == "admin" and _int(form.get("clinic_id")):
        p.clinic_id = _int(form.get("clinic_id"))
    db.commit()
    flash(request, "Patient details saved.")
    return redirect(f"/patients/{pid}")


# ----------------------------------------------------------------------------
# appointments
# ----------------------------------------------------------------------------
@app.get("/appointments")
def appointments(request: Request, day: str = "", status: str = "", doctor: str = "",
                 db=Depends(get_db), user: User = Depends(current_user)):
    q = scoped_appointments(db, user)
    d = _date(day)
    if d:
        start = datetime.combine(d, datetime.min.time())
        q = q.filter(Appointment.scheduled_at >= start, Appointment.scheduled_at < start + timedelta(days=1))
    if status in APPT_STATUSES:
        q = q.filter(Appointment.status == status)
    doc_id = _int(doctor)
    if doctor == "me":
        doc_id = user.id
    if doc_id:
        q = q.filter(Appointment.doctor_id == doc_id)
    rows = q.order_by(Appointment.scheduled_at.desc()).limit(300).all()
    return render(request, "appointments.html", user, appts=rows, day=day, status=status,
                  doctor=doctor, doctors=doctors_for(db, user), statuses=APPT_STATUSES)


@app.get("/appointments/new")
def appointment_new(request: Request, patient_id: int | None = None, db=Depends(get_db),
                    user: User = Depends(current_user)):
    p = get_patient(db, user, patient_id) if patient_id else None
    pts = [] if p else scoped_patients(db, user).order_by(Patient.last_name).all()
    default_time = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    return render(request, "appointment_form.html", user, p=p, patients=pts,
                  doctors=doctors_for(db, user, p.clinic_id if p else None),
                  default_time=default_time.strftime("%Y-%m-%dT%H:%M"))


@app.post("/appointments/new")
async def appointment_create(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    form = await request.form()
    pid = _int(form.get("patient_id"))
    back = f"/appointments/new?patient_id={pid}" if pid else "/appointments/new"
    try:
        when = datetime.fromisoformat(_s(form, "scheduled_at"))
    except ValueError:
        flash(request, "Choose a valid date and time.", "error")
        return redirect(back)
    if not pid:
        flash(request, "Choose a patient.", "error")
        return redirect(back)
    p = get_patient(db, user, pid)
    doc = _int(form.get("doctor_id"))
    if doc and db.query(Appointment).filter(Appointment.doctor_id == doc, Appointment.status == "scheduled",
                                            Appointment.scheduled_at > when - timedelta(minutes=15),
                                            Appointment.scheduled_at < when + timedelta(minutes=15)).first():
        flash(request, "That doctor already has an appointment within 15 minutes of this time.", "error")
        return redirect(back)
    a = Appointment(patient_id=p.id, doctor_id=doc, scheduled_at=when, reason=_s(form, "reason"))
    notify.mark_past_reminders(a)
    db.add(a)
    db.commit()
    sent = notify.notify_patient(db, a, "confirmation")
    flash(request, f"Appointment booked for {p.full_name} on {_fmt_dt(when)}."
                   + (f" Confirmation emailed to {p.email}." if sent else ""))
    return redirect(f"/patients/{p.id}?tab=appointments")


@app.get("/appointments/{aid}")
def appointment_detail(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    a = db.get(Appointment, aid)
    if not a:
        raise HTTPException(404)
    get_patient(db, user, a.patient_id)
    rx = db.query(Prescription).filter(Prescription.appointment_id == a.id).order_by(Prescription.id).all()
    emails = db.query(Notification).filter(Notification.appointment_id == a.id,
                                           Notification.patient_id == a.patient_id,
                                           Notification.kind != "new-booking").order_by(Notification.id.desc()).all()
    return render(request, "appointment_detail.html", user, a=a, statuses=APPT_STATUSES, rx=rx, emails=emails,
                  doctors=doctors_for(db, user, a.patient.clinic_id))


@app.post("/appointments/{aid}")
async def appointment_update(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    a = db.get(Appointment, aid)
    if not a:
        raise HTTPException(404)
    get_patient(db, user, a.patient_id)
    form = await request.form()
    old_status, old_time = a.status, a.scheduled_at
    if _s(form, "status") in APPT_STATUSES:
        a.status = _s(form, "status")
    if "notes" in form:
        a.notes = _s(form, "notes")
    if "reason" in form:
        a.reason = _s(form, "reason")
    if "doctor_id" in form:
        a.doctor_id = _int(form.get("doctor_id"))
    if _s(form, "scheduled_at"):
        try:
            a.scheduled_at = datetime.fromisoformat(_s(form, "scheduled_at"))
        except ValueError:
            pass
    a.acknowledged = 1
    emailed = None
    if a.status == "cancelled" and old_status != "cancelled":
        db.commit()
        emailed = notify.notify_patient(db, a, "cancelled")
    elif a.status == "scheduled" and (a.scheduled_at != old_time or old_status != "scheduled"):
        notify.mark_past_reminders(a)
        db.commit()
        emailed = notify.notify_patient(db, a, "rescheduled")
    db.commit()
    flash(request, "Appointment updated." + (f" The patient was emailed at {a.patient.email}." if emailed else ""))
    return redirect(f"/appointments/{aid}")


@app.post("/appointments/{aid}/remind")
def appointment_remind(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    a = db.get(Appointment, aid)
    if not a:
        raise HTTPException(404)
    get_patient(db, user, a.patient_id)
    if not a.patient.email:
        flash(request, "This patient has no email address. Add one on the patient's details.", "error")
    elif a.status != "scheduled" or a.scheduled_at < datetime.now():
        flash(request, "Reminders can only be sent for upcoming scheduled appointments.", "error")
    else:
        hours = max(1, int((a.scheduled_at - datetime.now()).total_seconds() // 3600))
        subject, body = notify.appointment_message(a, "reminder", hours)
        notify.queue(db, to=a.patient.email, subject=subject, body=body, kind="reminder",
                     patient_id=a.patient_id, appointment_id=a.id)
        flash(request, f"Reminder emailed to {a.patient.email}.")
    return redirect(f"/appointments/{aid}")


@app.post("/appointments/{aid}/acknowledge")
def appointment_ack(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    a = db.get(Appointment, aid)
    if not a:
        raise HTTPException(404)
    get_patient(db, user, a.patient_id)
    a.acknowledged = 1
    db.commit()
    return redirect(request.headers.get("referer") or "/")


# ----------------------------------------------------------------------------
# diagnoses
# ----------------------------------------------------------------------------
@app.get("/patients/{pid}/diagnoses/new")
def diagnosis_new(pid: int, request: Request, condition: str = "", disease_id: str = "",
                  analysis_id: int | None = None, appointment_id: int | None = None,
                  db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    return render(request, "diagnosis_form.html", user, p=p, condition=condition, disease_id=disease_id,
                  analysis_id=analysis_id, appointment_id=appointment_id,
                  appts=sorted(p.appointments, key=lambda a: a.scheduled_at, reverse=True))


@app.post("/patients/{pid}/diagnoses/new")
async def diagnosis_create(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    form = await request.form()
    if not _s(form, "condition"):
        flash(request, "Condition is required.", "error")
        return redirect(f"/patients/{pid}/diagnoses/new")
    d = Diagnosis(patient_id=p.id, doctor_id=user.id if user.role == "doctor" else None,
                  condition=_s(form, "condition"), disease_id=_s(form, "disease_id"),
                  notes=_s(form, "notes"), diagnosed_on=_date(form.get("diagnosed_on")) or date.today(),
                  appointment_id=_int(form.get("appointment_id")), analysis_id=_int(form.get("analysis_id")))
    db.add(d)
    db.commit()
    flash(request, f"Diagnosis “{d.condition}” added to {p.full_name}'s record.")
    return redirect(f"/patients/{pid}?tab=diagnoses")


@app.post("/diagnoses/{did}/status")
async def diagnosis_status(did: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    d = db.get(Diagnosis, did)
    get_patient(db, user, d.patient_id)
    form = await request.form()
    if _s(form, "status") == "resolved":
        d.status, d.resolved_on = "resolved", date.today()
    else:
        d.status, d.resolved_on = "active", None
    db.commit()
    flash(request, f"Diagnosis marked {d.status}.")
    return redirect(f"/patients/{d.patient_id}?tab=diagnoses")


# ----------------------------------------------------------------------------
# medicines (prescriptions)
# ----------------------------------------------------------------------------
@app.get("/patients/{pid}/medicines/new")
def medicine_new(pid: int, request: Request, appointment_id: int | None = None,
                 diagnosis_id: int | None = None, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    return render(request, "medicine_form.html", user, p=p, appointment_id=appointment_id,
                  diagnosis_id=diagnosis_id,
                  active_dx=[d for d in p.diagnoses if d.status == "active"],
                  appts=sorted(p.appointments, key=lambda a: a.scheduled_at, reverse=True))


@app.post("/patients/{pid}/medicines/new")
async def medicine_create(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    form = await request.form()
    cols = {k: [str(v).strip() for v in form.getlist(k)]
            for k in ("medicine", "dosage", "frequency", "duration", "instructions")}
    dx_id = _int(form.get("diagnosis_id"))
    if dx_id and not any(d.id == dx_id for d in p.diagnoses):
        dx_id = None
    appt_id = _int(form.get("appointment_id"))
    if appt_id and not any(a.id == appt_id for a in p.appointments):
        appt_id = None

    def col(k, i):
        return cols[k][i] if i < len(cols[k]) else ""

    added = []
    for i, name in enumerate(cols["medicine"]):
        if not name:
            continue
        db.add(Prescription(patient_id=p.id, doctor_id=user.id if user.role == "doctor" else None,
                            appointment_id=appt_id, diagnosis_id=dx_id, medicine=name,
                            dosage=col("dosage", i), frequency=col("frequency", i),
                            duration=col("duration", i), instructions=col("instructions", i)))
        added.append(name)
    if not added:
        flash(request, "Enter at least one medicine.", "error")
        return redirect(f"/patients/{pid}/medicines/new")
    db.commit()
    flash(request, f"Prescribed {', '.join(added)} for {p.full_name}.")
    if appt_id and form.get("back") == "appointment":
        return redirect(f"/appointments/{appt_id}")
    return redirect(f"/patients/{pid}?tab=medicines")


@app.post("/prescriptions/{rid}/status")
async def prescription_status(rid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    r = db.get(Prescription, rid)
    if not r:
        raise HTTPException(404)
    get_patient(db, user, r.patient_id)
    form = await request.form()
    if _s(form, "status") == "stopped":
        r.status, r.stopped_on = "stopped", date.today()
    else:
        r.status, r.stopped_on = "active", None
    db.commit()
    flash(request, f"{r.medicine} marked {r.status}.")
    return redirect(f"/patients/{r.patient_id}?tab=medicines")


# ----------------------------------------------------------------------------
# reports (Excel export)
# ----------------------------------------------------------------------------
REPORT_RANGES = [("today", "Today"), ("2d", "Last 2 days"), ("7d", "Last 7 days"),
                 ("30d", "Last 30 days"), ("custom", "Custom dates")]


def _report_args(request: Request, user: User):
    qp = request.query_params
    range_ = qp.get("range", "today")
    if range_ not in dict(REPORT_RANGES):
        range_ = "today"
    t = date.today()
    days = {"today": 1, "2d": 2, "7d": 7, "30d": 30}.get(range_)
    if days:
        d0, d1 = t - timedelta(days=days - 1), t
    else:
        d0, d1 = _date(qp.get("start")) or t, _date(qp.get("end")) or t
        if d0 > d1:
            d0, d1 = d1, d0
    doctor = qp.get("doctor", "")
    return range_, d0, d1, doctor


def _report_rows(db, user: User, d0: date, d1: date, doctor: str):
    q = db.query(Prescription).join(Patient).filter(
        Prescription.prescribed_at >= datetime.combine(d0, datetime.min.time()),
        Prescription.prescribed_at < datetime.combine(d1 + timedelta(days=1), datetime.min.time()))
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    doc_id = user.id if doctor == "me" else _int(doctor)
    if doc_id:
        q = q.filter(Prescription.doctor_id == doc_id)
    out = []
    for r in q.order_by(Prescription.prescribed_at.desc()).all():
        p = r.patient
        disease = r.diagnosis.condition if r.diagnosis else "; ".join(
            d.condition for d in p.diagnoses if d.status == "active")
        out.append({"r": r, "p": p, "disease": disease or "—"})
    return out


@app.get("/reports")
def reports(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    range_, d0, d1, doctor = _report_args(request, user)
    rows = _report_rows(db, user, d0, d1, doctor)
    return render(request, "reports.html", user, rows=rows, range=range_, ranges=REPORT_RANGES,
                  d0=d0, d1=d1, doctor=doctor, doctors=doctors_for(db, user),
                  n_patients=len({x["p"].id for x in rows}), query=request.url.query)


@app.get("/reports/export.xlsx")
def reports_export(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    range_, d0, d1, doctor = _report_args(request, user)
    rows = _report_rows(db, user, d0, d1, doctor)
    now = datetime.now()

    wb = Workbook()
    ws = wb.active
    ws.title = "Medicines"
    headers = ["Date", "Time", "Patient ID", "Patient name", "Age", "Sex", "Disease / diagnosis",
               "Medicine", "Dosage", "Frequency", "Duration", "Instructions", "Status",
               "Prescribed by", "Clinic"]
    widths = [13, 10, 11, 22, 6, 8, 32, 30, 12, 20, 12, 30, 10, 20, 24]
    scope = "All clinics" if user.role == "admin" else (user.clinic.label if user.clinic else "")
    period = f"{d0:%d %b %Y}" if d0 == d1 else f"{d0:%d %b %Y} to {d1:%d %b %Y}"
    ws.append(["MediPurse: medicines given"])
    ws.append([f"Period: {period}   |   Clinic: {scope}"])
    ws.append([f"Generated: {now:%d %b %Y, %I:%M %p} by {user.display}   |   "
               f"{len(rows)} prescriptions, {len({x['p'].id for x in rows})} patients"])
    ws.append([])
    ws.append(headers)
    ws["A1"].font = Font(size=14, bold=True)
    ws["A2"].font = Font(color="555555")
    ws["A3"].font = Font(color="555555")
    head_row = 5
    thin = Side(style="thin", color="D4D4D8")
    for c in ws[head_row]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2563EB")
        c.alignment = Alignment(vertical="center")
    for x in rows:
        r, p = x["r"], x["p"]
        ws.append([r.prescribed_at.date(), r.prescribed_at.time().replace(second=0, microsecond=0),
                   p.mrn, p.full_name, p.age, p.sex, x["disease"], r.medicine, r.dosage,
                   r.frequency, r.duration, r.instructions, r.status,
                   r.doctor.display if r.doctor else "", p.clinic.name])
        row = ws.max_row
        ws.cell(row, 1).number_format = "DD-MMM-YYYY"
        ws.cell(row, 2).number_format = "hh:mm AM/PM"
        for c in ws[row]:
            c.border = Border(bottom=thin)
            c.alignment = Alignment(vertical="top", wrap_text=True)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(head_row + 1, 1)
    if rows:
        ws.auto_filter.ref = f"A{head_row}:{get_column_letter(len(headers))}{ws.max_row}"

    buf = BytesIO()
    wb.save(buf)
    fname = f"medicines_{d0:%Y-%m-%d}_to_{d1:%Y-%m-%d}_generated_{now:%Y-%m-%d_%H%M}.xlsx"
    return Response(buf.getvalue(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ----------------------------------------------------------------------------
# email log
# ----------------------------------------------------------------------------
@app.get("/notifications")
def notifications(request: Request, kind: str = "", status: str = "", db=Depends(get_db),
                  user: User = Depends(current_user)):
    q = db.query(Notification).outerjoin(Patient, Notification.patient_id == Patient.id)
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    if kind:
        q = q.filter(Notification.kind == kind)
    if status:
        q = q.filter(Notification.status == status)
    rows = q.order_by(Notification.id.desc()).limit(300).all()
    return render(request, "notifications.html", user, rows=rows, kind=kind, status=status,
                  smtp_on=notify.smtp_configured(), outbox=str(notify.outbox_path()),
                  hours=notify.reminder_hours())


@app.post("/notifications/{nid}/retry")
def notification_retry(nid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    n = db.get(Notification, nid)
    if not n or (user.role != "admin" and (not n.patient or n.patient.clinic_id != user.clinic_id)):
        raise HTTPException(404)
    n.status, n.attempts, n.error = "queued", 0, ""
    db.commit()
    notify._wake.set()
    flash(request, "Email queued again.")
    return redirect("/notifications")


@app.get("/notifications/{nid}")
def notification_view(nid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    n = db.get(Notification, nid)
    if not n or (user.role != "admin" and (not n.patient or n.patient.clinic_id != user.clinic_id)):
        raise HTTPException(404)
    return render(request, "notification_detail.html", user, n=n)


# ----------------------------------------------------------------------------
# feedback
# ----------------------------------------------------------------------------
@app.get("/feedback")
def feedback_list(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    q = db.query(Feedback).join(Patient)
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    rows = q.order_by(Feedback.created_at.desc()).all()
    avg = round(sum(f.rating for f in rows) / len(rows), 1) if rows else None
    return render(request, "feedback.html", user, rows=rows, avg=avg)


@app.get("/patients/{pid}/feedback/new")
def feedback_new(pid: int, request: Request, appointment_id: int | None = None,
                 db=Depends(get_db), user: User = Depends(current_user)):
    p = get_patient(db, user, pid)
    return render(request, "feedback_form.html", user, p=p, appointment_id=appointment_id,
                  appts=sorted(p.appointments, key=lambda a: a.scheduled_at, reverse=True))


@app.post("/patients/{pid}/feedback/new")
async def feedback_create(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    p = get_patient(db, user, pid)
    form = await request.form()
    rating = _int(form.get("rating"))
    if not rating or not 1 <= rating <= 5:
        flash(request, "Choose a rating from 1 to 5.", "error")
        return redirect(f"/patients/{pid}/feedback/new")
    db.add(Feedback(patient_id=p.id, rating=rating, comments=_s(form, "comments"),
                    appointment_id=_int(form.get("appointment_id")), recorded_by=user.id))
    db.commit()
    flash(request, "Feedback recorded. Thank you.")
    return redirect(f"/patients/{pid}?tab=feedback")


# ----------------------------------------------------------------------------
# AI diagnosis assistant
# ----------------------------------------------------------------------------
@app.get("/ai")
def ai_home(request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    q = db.query(AIAnalysis).join(Patient)
    if user.role != "admin":
        q = q.filter(Patient.clinic_id == user.clinic_id)
    rows = q.order_by(AIAnalysis.created_at.desc()).limit(50).all()
    return render(request, "ai_home.html", user, rows=rows,
                  patients=scoped_patients(db, user).order_by(Patient.last_name).all())


@app.get("/api/ai-status")
def ai_status():
    return {"status": AI.status, "llm": AI.llm_configured}


def _default_case_text(p: Patient, appointment: Appointment | None) -> str:
    parts = []
    if p.age is not None:
        sex = {"Male": "man" if p.age >= 18 else "boy",
               "Female": "woman" if p.age >= 18 else "girl"}.get(p.sex, "patient")
        parts.append(f"A {p.age}-year-old {sex}")
    if appointment and appointment.reason:
        parts.append(f"presenting with {appointment.reason.lower()}")
    text = " ".join(parts)
    return (text + ". ") if text else ""


@app.get("/patients/{pid}/ai/new")
def ai_new(pid: int, request: Request, appointment_id: int | None = None,
           db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    appt = db.get(Appointment, appointment_id) if appointment_id else None
    return render(request, "ai_new.html", user, p=p, case_text=_default_case_text(p, appt))


@app.post("/patients/{pid}/ai/new")
async def ai_create(pid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    p = get_patient(db, user, pid)
    form = await request.form()
    text = _s(form, "case_text")
    if len(text.split()) < 5:
        flash(request, "Describe the case in a few sentences: symptoms, signs, test results, history.", "error")
        return redirect(f"/patients/{pid}/ai/new")
    if not AI.ready:
        flash(request, "The AI model is still loading (about a minute after start). Try again shortly.", "error")
        return redirect(f"/patients/{pid}/ai/new")
    ranking = AI.rank(text)
    an = AIAnalysis(patient_id=p.id, doctor_id=user.id, case_text=text, ranking_json=json.dumps(ranking))
    db.add(an)
    db.commit()
    return redirect(f"/ai/{an.id}")


def _get_analysis(db, user, aid) -> AIAnalysis:
    an = db.get(AIAnalysis, aid)
    if not an:
        raise HTTPException(404)
    get_patient(db, user, an.patient_id)
    return an


@app.get("/ai/{aid}")
def ai_result(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    an = _get_analysis(db, user, aid)
    ranking = json.loads(an.ranking_json or "{}")
    explain = json.loads(an.explain_json) if an.explain_json else None
    saved = {d.disease_id for d in an.patient.diagnoses if d.disease_id}
    return render(request, "ai_result.html", user, an=an, p=an.patient, ranking=ranking,
                  explain=explain, saved=saved)


def _run_explain(aid: int) -> None:
    db = SessionLocal()
    try:
        an = db.get(AIAnalysis, aid)

        def progress(stage):
            an.explain_stage = stage
            db.commit()

        try:
            result = AI.explain(an.case_text, case_id=f"A{aid}", progress=progress)
            an.explain_json = json.dumps(result)
            an.explain_status, an.explain_stage, an.explain_error = "done", "", ""
        except Exception as exc:
            an.explain_status, an.explain_error = "error", str(exc)[:1000]
        db.commit()
    finally:
        db.close()


@app.post("/ai/{aid}/explain")
def ai_explain(aid: int, request: Request, db=Depends(get_db), user: User = Depends(current_user)):
    require(user, "doctor", "admin")
    an = _get_analysis(db, user, aid)
    if not AI.llm_configured:
        flash(request, "AI explanations are not available right now.", "error")
        return redirect(f"/ai/{aid}")
    if an.explain_status != "running":
        an.explain_status, an.explain_stage, an.explain_error = "running", "Starting", ""
        db.commit()
        threading.Thread(target=_run_explain, args=(aid,), daemon=True).start()
    return redirect(f"/ai/{aid}")


@app.get("/ai/{aid}/status")
def ai_explain_status(aid: int, db=Depends(get_db), user: User = Depends(current_user)):
    an = _get_analysis(db, user, aid)
    return JSONResponse({"status": an.explain_status, "stage": an.explain_stage})


# patient portal (registered last; it imports helpers from this module)
from .portal import router as portal_router  # noqa: E402

app.include_router(portal_router)
