"""Database models (SQLite via SQLAlchemy)."""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (Date, DateTime, ForeignKey, Integer, String, Text,
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
    patient: Mapped[Patient] = relationship(back_populates="prescriptions")
    doctor: Mapped[User | None] = relationship()
    appointment: Mapped[Appointment | None] = relationship()
    diagnosis: Mapped[Diagnosis | None] = relationship()


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    rating: Mapped[int] = mapped_column(Integer)
    comments: Mapped[str] = mapped_column(Text, default="")
    recorded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="feedback")
    appointment: Mapped[Appointment | None] = relationship()


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient: Mapped[Patient] = relationship(back_populates="analyses")
    doctor: Mapped[User | None] = relationship()


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


def count(db, model) -> int:
    return db.query(func.count(model.id)).scalar() or 0
