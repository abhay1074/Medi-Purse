"""Database models (SQLite via SQLAlchemy)."""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, func)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("HMS_DB", ROOT / "hospital.db"))

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


ROLES = {
    "admin": "Registrar / Admin",
    "doctor": "Doctor",
    "receptionist": "Receptionist",
    "pharmacist": "Pharmacist",
}


class Hospital(Base):
    __tablename__ = "hospitals"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    registration_no: Mapped[str] = mapped_column(String(100), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    city: Mapped[str] = mapped_column(String(100), default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    email: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    clinics: Mapped[list["Clinic"]] = relationship(back_populates="hospital")


class Clinic(Base):
    __tablename__ = "clinics"
    id: Mapped[int] = mapped_column(primary_key=True)
    hospital_id: Mapped[int | None] = mapped_column(ForeignKey("hospitals.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    specialty: Mapped[str] = mapped_column(String(100), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    hospital: Mapped[Hospital | None] = relationship(back_populates="clinics")
    staff: Mapped[list["User"]] = relationship(back_populates="clinic")
    patients: Mapped[list["Patient"]] = relationship(back_populates="clinic")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.hospital.name})" if self.hospital else self.name


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20))
    specialty: Mapped[str] = mapped_column(String(100), default="")
    clinic_id: Mapped[int | None] = mapped_column(ForeignKey("clinics.id"), nullable=True)
    active: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    clinic: Mapped[Clinic | None] = relationship(back_populates="staff")

    @property
    def role_label(self) -> str:
        return ROLES.get(self.role, self.role)

    @property
    def display(self) -> str:
        return f"Dr. {self.name}" if self.role == "doctor" else self.name


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100), default="")
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str] = mapped_column(String(20), default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    email: Mapped[str] = mapped_column(String(200), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    blood_group: Mapped[str] = mapped_column(String(10), default="")
    allergies: Mapped[str] = mapped_column(Text, default="")
    emergency_contact: Mapped[str] = mapped_column(String(200), default="")
    # patient portal
    password_hash: Mapped[str] = mapped_column(String(300), default="")
    portal_token: Mapped[str] = mapped_column(String(100), default="")
    token_expires: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    email_reminders: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    clinic: Mapped[Clinic] = relationship(back_populates="patients")
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="patient")
    diagnoses: Mapped[list["Diagnosis"]] = relationship(back_populates="patient")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="patient")
    analyses: Mapped[list["AIAnalysis"]] = relationship(back_populates="patient")
    prescriptions: Mapped[list["Prescription"]] = relationship(back_populates="patient")

    @property
    def mrn(self) -> str:
        return f"P-{self.id:06d}"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def age(self) -> int | None:
        if not self.dob:
            return None
        t = date.today()
        return t.year - self.dob.year - ((t.month, t.day) < (self.dob.month, self.dob.day))


APPT_STATUSES = ["scheduled", "completed", "cancelled", "no-show"]


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime)
    reason: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    notes: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(20), default="staff")  # staff | online
    acknowledged: Mapped[int] = mapped_column(Integer, default=1)  # 0 = new online booking
    reminders_sent: Mapped[str] = mapped_column(String(50), default="")  # e.g. "24,2"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="appointments")
    doctor: Mapped[User | None] = relationship()


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    analysis_id: Mapped[int | None] = mapped_column(ForeignKey("ai_analyses.id"), nullable=True)
    condition: Mapped[str] = mapped_column(String(300))
    disease_id: Mapped[str] = mapped_column(String(50), default="")
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | resolved
    notes: Mapped[str] = mapped_column(Text, default="")
    diagnosed_on: Mapped[date] = mapped_column(Date, default=date.today)
    resolved_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="diagnoses")
    doctor: Mapped[User | None] = relationship()


class Prescription(Base):
    """A medicine prescribed to a patient by a doctor."""
    __tablename__ = "prescriptions"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    diagnosis_id: Mapped[int | None] = mapped_column(ForeignKey("diagnoses.id"), nullable=True)
    medicine: Mapped[str] = mapped_column(String(200))
    dosage: Mapped[str] = mapped_column(String(100), default="")
    frequency: Mapped[str] = mapped_column(String(100), default="")
    duration: Mapped[str] = mapped_column(String(100), default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | stopped
    prescribed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    stopped_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    # pharmacy: what the clinic gave from its own stock; the rest is bought elsewhere
    drug_id: Mapped[int | None] = mapped_column(ForeignKey("drugs.id"), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)      # units needed for the course
    dispensed: Mapped[int] = mapped_column(Integer, default=0)     # units given from clinic stock
    supply: Mapped[str] = mapped_column(String(20), default="")    # clinic | partial | outside | "" (older)
    # dispensing by the pharmacist or receptionist (stock is deducted then, not when prescribed)
    dispense_status: Mapped[str] = mapped_column(String(20), default="")  # pending | done | "" (older rows)
    rx_group: Mapped[str] = mapped_column(String(32), default="", index=True)  # lines saved together
    dispensed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    dispensed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    patient: Mapped[Patient] = relationship(back_populates="prescriptions")
    drug: Mapped["Drug | None"] = relationship()
    doctor: Mapped[User | None] = relationship(foreign_keys=[doctor_id])
    dispensed_by: Mapped[User | None] = relationship(foreign_keys=[dispensed_by_id])
    appointment: Mapped[Appointment | None] = relationship()
    diagnosis: Mapped[Diagnosis | None] = relationship()


class Drug(Base):
    """A medicine held in a clinic's pharmacy stock."""
    __tablename__ = "drugs"
    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"), index=True)
    name: Mapped[str] = mapped_column(String(150))            # generic name, e.g. Paracetamol
    strength: Mapped[str] = mapped_column(String(60), default="")   # 500 mg, 5 mg/5 ml
    form: Mapped[str] = mapped_column(String(30), default="tablet")  # tablet | capsule | syrup | ...
    unit: Mapped[str] = mapped_column(String(30), default="tablet")  # what stock is counted in
    stock: Mapped[int] = mapped_column(Integer, default=0)
    reorder_level: Mapped[int] = mapped_column(Integer, default=20)
    batch: Mapped[str] = mapped_column(String(60), default="")
    expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    price: Mapped[float] = mapped_column(Float, default=0.0)   # selling price per unit (Rs)
    active: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    clinic: Mapped[Clinic] = relationship()

    @property
    def label(self) -> str:
        return " ".join(x for x in (self.name, self.strength, self.form) if x)

    @property
    def level(self) -> str:
        if self.stock <= 0:
            return "out"
        return "low" if self.stock <= self.reorder_level else "ok"


class StockMove(Base):
    """Every change to a drug's stock: received, dispensed on a prescription, adjusted."""
    __tablename__ = "stock_moves"
    id: Mapped[int] = mapped_column(primary_key=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"), index=True)
    change: Mapped[int] = mapped_column(Integer)
    balance: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(20))  # received | dispensed | adjusted | expired | added
    note: Mapped[str] = mapped_column(String(300), default="")
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    prescription_id: Mapped[int | None] = mapped_column(ForeignKey("prescriptions.id"), nullable=True)
    sale_id: Mapped[int | None] = mapped_column(ForeignKey("sales.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    drug: Mapped[Drug] = relationship()
    user: Mapped[User | None] = relationship()


class Sale(Base):
    """A counter sale by the pharmacist (walk-in customer or an outside prescription)."""
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"), index=True)
    pharmacist_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    customer_name: Mapped[str] = mapped_column(String(200), default="")
    customer_phone: Mapped[str] = mapped_column(String(50), default="")
    prescribed_by: Mapped[str] = mapped_column(String(200), default="")  # outside doctor / hospital, if any
    payment: Mapped[str] = mapped_column(String(20), default="cash")      # cash | upi | card
    total: Mapped[float] = mapped_column(Float, default=0.0)
    note: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    clinic: Mapped[Clinic] = relationship()
    pharmacist: Mapped[User | None] = relationship()
    patient: Mapped[Patient | None] = relationship()
    items: Mapped[list["SaleItem"]] = relationship(back_populates="sale", cascade="all, delete-orphan")


class SaleItem(Base):
    __tablename__ = "sale_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    sale_id: Mapped[int] = mapped_column(ForeignKey("sales.id"), index=True)
    drug_id: Mapped[int] = mapped_column(ForeignKey("drugs.id"))
    name: Mapped[str] = mapped_column(String(250))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0)
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    sale: Mapped[Sale] = relationship(back_populates="items")
    drug: Mapped[Drug] = relationship()


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    rating: Mapped[int] = mapped_column(Integer)
    comments: Mapped[str] = mapped_column(Text, default="")
    recorded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="staff")  # staff | patient (email link)
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="feedback")
    doctor: Mapped[User | None] = relationship(foreign_keys=[doctor_id])
    appointment: Mapped[Appointment | None] = relationship()


class FeedbackRequest(Base):
    """A one-time link emailed to a patient after a visit or diagnosis, asking for feedback."""
    __tablename__ = "feedback_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    diagnosis_id: Mapped[int | None] = mapped_column(ForeignKey("diagnoses.id"), nullable=True)
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    feedback_id: Mapped[int | None] = mapped_column(ForeignKey("feedback.id"), nullable=True)
    patient: Mapped[Patient] = relationship()
    appointment: Mapped[Appointment | None] = relationship()
    doctor: Mapped[User | None] = relationship()


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    case_text: Mapped[str] = mapped_column(Text)
    # ranking: instant local ranker; explanation: optional DeepSeek pass
    ranking_json: Mapped[str] = mapped_column(Text, default="")
    explain_status: Mapped[str] = mapped_column(String(20), default="none")  # none|running|done|error
    explain_stage: Mapped[str] = mapped_column(String(50), default="")
    explain_json: Mapped[str] = mapped_column(Text, default="")
    explain_error: Mapped[str] = mapped_column(Text, default="")
    # realist agent (rules immediately, refined by the LLM when available)
    realist_json: Mapped[str] = mapped_column(Text, default="")
    realist_status: Mapped[str] = mapped_column(String(20), default="none")  # none|rules|running|done
    symptoms_json: Mapped[str] = mapped_column(Text, default="")  # symptoms picked in the search box
    history_id: Mapped[int | None] = mapped_column(ForeignKey("history_intakes.id"), nullable=True)
    image_ids: Mapped[str] = mapped_column(Text, default="")  # JSON list of PatientImage ids (max 4)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="analyses")
    doctor: Mapped[User | None] = relationship()


class PatientImage(Base):
    """A clinical image in the patient's record (files live in uploads/, see app/images.py)."""
    __tablename__ = "patient_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    source: Mapped[str] = mapped_column(String(20), default="staff")  # staff | patient
    uploaded_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    intake_id: Mapped[int | None] = mapped_column(ForeignKey("history_intakes.id"), nullable=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    modality: Mapped[str] = mapped_column(String(30), default="clinical_photo")
    body_site: Mapped[str] = mapped_column(String(120), default="")
    finding: Mapped[str] = mapped_column(Text, default="")   # what the image shows, in words
    finding_source: Mapped[str] = mapped_column(String(20), default="clinician")  # clinician | patient
    taken_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    original_name: Mapped[str] = mapped_column(String(255), default="")
    stored_name: Mapped[str] = mapped_column(String(80))
    thumb_name: Mapped[str] = mapped_column(String(80))
    mime: Mapped[str] = mapped_column(String(40))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    caption: Mapped[str] = mapped_column(Text, default="")
    caption_model: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    patient: Mapped[Patient] = relationship()
    uploaded_by: Mapped[User | None] = relationship(foreign_keys=[uploaded_by_id])


class HistoryIntake(Base):
    """A history-taking conversation between a patient and the intake agent."""
    __tablename__ = "history_intakes"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(20), default="guided")  # ai | guided
    status: Mapped[str] = mapped_column(String(20), default="in_progress")  # in_progress | completed
    step: Mapped[int] = mapped_column(Integer, default=0)
    messages_json: Mapped[str] = mapped_column(Text, default="[]")
    answers_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="")
    case_text: Mapped[str] = mapped_column(Text, default="")
    red_flags: Mapped[str] = mapped_column(Text, default="")
    reviewed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    patient: Mapped[Patient] = relationship()
    appointment: Mapped[Appointment | None] = relationship()


class Notification(Base):
    """An email sent (or queued) to a patient or staff member."""
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(30))  # confirmation|reminder|rescheduled|cancelled|invite|password|new-booking
    to_email: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|sent|saved|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    patient: Mapped[Patient | None] = relationship()
    appointment: Mapped[Appointment | None] = relationship()


# ----------------------------------------------------------------------------
def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${h}"


def check_password(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
    except ValueError:
        return False
    calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return secrets.compare_digest(calc, h)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Tiny migration: add columns introduced after a database was created (SQLite)."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                ddl = f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'
                if default is not None:
                    ddl += f" DEFAULT {default!r}" if isinstance(default, str) else f" DEFAULT {default}"
                conn.execute(text(ddl))


def count(db, model) -> int:
    return db.query(func.count(model.id)).scalar() or 0
