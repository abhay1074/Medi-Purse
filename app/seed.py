"""Demo data, created on first start when the database is empty.

These are local test accounts only. Change or delete them before any real use.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .db import (Appointment, Clinic, Diagnosis, Feedback, Hospital, Patient,
                 SessionLocal, User, hash_password)

DEMO_PASSWORD = "demo1234"
DEMO_ACCOUNTS = [
    ("admin@medipurse.test", "Registrar / Admin"),
    ("dr.mehta@medipurse.test", "Doctor, Genetics Clinic"),
    ("dr.rao@medipurse.test", "Doctor, Neurology Clinic"),
    ("reception@medipurse.test", "Receptionist, Genetics Clinic"),
]


def seed_if_empty() -> None:
    db = SessionLocal()
    try:
        if db.query(User).count():
            return
        h = Hospital(name="City General Hospital", registration_no="HOSP-2026-001",
                     address="12 Station Road", city="Raipur", phone="0771-400100",
                     email="contact@citygeneral.test")
        db.add(h)
        db.flush()
        gen = Clinic(hospital_id=h.id, name="Genetics Clinic", specialty="Medical genetics",
                     address="Block B, 2nd floor", phone="0771-400121")
        neuro = Clinic(hospital_id=h.id, name="Neurology Clinic", specialty="Neurology",
                       address="Block A, 1st floor", phone="0771-400131")
        family = Clinic(name="Sunrise Family Clinic", specialty="General practice",
                        address="44 Lake View", phone="0771-555010")
        db.add_all([gen, neuro, family])
        db.flush()

        pw = hash_password(DEMO_PASSWORD)
        admin = User(name="Registrar", email="admin@medipurse.test", role="admin", password_hash=pw)
        mehta = User(name="Anita Mehta", email="dr.mehta@medipurse.test", role="doctor",
                     specialty="Clinical geneticist", clinic_id=gen.id, password_hash=pw)
        rao = User(name="Vikram Rao", email="dr.rao@medipurse.test", role="doctor",
                   specialty="Neurologist", clinic_id=neuro.id, password_hash=pw)
        rec = User(name="Priya Sharma", email="reception@medipurse.test", role="receptionist",
                   clinic_id=gen.id, password_hash=pw)
        db.add_all([admin, mehta, rao, rec])
        db.flush()

        def pat(clinic, first, last, dob, sex, phone, **kw):
            p = Patient(clinic_id=clinic.id, first_name=first, last_name=last, dob=dob, sex=sex,
                        phone=phone, **kw)
            db.add(p)
            db.flush()
            return p

        p1 = pat(gen, "Sunita", "Verma", date(1980, 3, 14), "Female", "98260 11111",
                 blood_group="B+", allergies="Penicillin", emergency_contact="Rakesh Verma, 98260 11112")
        p2 = pat(gen, "Arjun", "Patel", date(2022, 7, 2), "Male", "98260 22222",
                 blood_group="O+", emergency_contact="Meera Patel (mother), 98260 22223")
        p3 = pat(gen, "Kavya", "Nair", date(1995, 11, 30), "Female", "98260 33333", blood_group="A+")
        p4 = pat(neuro, "Rohan", "Gupta", date(1968, 1, 9), "Male", "98260 44444", blood_group="AB+")
        p5 = pat(neuro, "Fatima", "Khan", date(2010, 5, 21), "Female", "98260 55555",
                 emergency_contact="Imran Khan (father), 98260 55556")
        pat(family, "Deepak", "Singh", date(1975, 8, 17), "Male", "98260 66666")

        now = datetime.now().replace(second=0, microsecond=0)
        today10 = now.replace(hour=10, minute=0)

        def appt(p, doc, when, reason, status="scheduled", notes=""):
            a = Appointment(patient_id=p.id, doctor_id=doc.id, scheduled_at=when, reason=reason,
                            status=status, notes=notes)
            db.add(a)
            db.flush()
            return a

        a1 = appt(p1, mehta, today10 - timedelta(days=60), "Blurred central vision",
                  "completed", "Fundus: angioid streaks both eyes. Referred for skin biopsy.")
        appt(p1, mehta, today10 + timedelta(hours=1), "Follow-up: biopsy results")
        appt(p2, mehta, today10 + timedelta(hours=2, minutes=30),
             "Recurrent jerking of left arm, developmental delay")
        a4 = appt(p3, mehta, today10 - timedelta(days=20), "Joint pain and easy bruising",
                  "completed", "Skin hyperextensible; Beighton score 7/9.")
        appt(p3, mehta, today10 + timedelta(days=7), "Genetic counselling")
        a6 = appt(p4, rao, today10 - timedelta(days=30), "Tremor in both hands", "completed",
                  "Postural tremor, no rigidity.")
        appt(p4, rao, today10 + timedelta(days=2), "Tremor review")
        appt(p5, rao, today10 - timedelta(days=3), "Headaches", "no-show")

        db.add_all([
            Diagnosis(patient_id=p1.id, doctor_id=mehta.id, appointment_id=a1.id,
                      condition="Pseudoxanthoma elasticum", disease_id="758", status="active",
                      notes="Suspected; awaiting biopsy confirmation.",
                      diagnosed_on=a1.scheduled_at.date()),
            Diagnosis(patient_id=p1.id, doctor_id=mehta.id, condition="Iron-deficiency anaemia",
                      status="resolved", diagnosed_on=date(2025, 2, 1), resolved_on=date(2025, 6, 1)),
            Diagnosis(patient_id=p3.id, doctor_id=mehta.id, appointment_id=a4.id,
                      condition="Hypermobile Ehlers-Danlos syndrome",
                      status="active", diagnosed_on=a4.scheduled_at.date()),
            Diagnosis(patient_id=p4.id, doctor_id=rao.id, appointment_id=a6.id,
                      condition="Essential tremor", status="active",
                      diagnosed_on=a6.scheduled_at.date()),
        ])
        db.add_all([
            Feedback(patient_id=p1.id, appointment_id=a1.id, rating=5,
                     comments="Doctor explained everything clearly.", recorded_by=rec.id),
            Feedback(patient_id=p3.id, appointment_id=a4.id, rating=4,
                     comments="Long wait at reception, but good consultation.", recorded_by=rec.id),
        ])
        db.commit()
    finally:
        db.close()
