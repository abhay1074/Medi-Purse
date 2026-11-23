"""Demo data, created on first start when the database is empty.

These are local test accounts only. Change or delete them before any real use.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .db import (Appointment, Clinic, Diagnosis, Feedback, Hospital, Patient,
                 Prescription, SessionLocal, User, hash_password)

DEMO_PASSWORD = "demo1234"
DEMO_ACCOUNTS = [
    ("admin@medipurse.test", "Registrar / Admin"),
    ("dr.mehta@medipurse.test", "Doctor, Genetics Clinic"),
    ("dr.rao@medipurse.test", "Doctor, Neurology Clinic"),
    ("reception@medipurse.test", "Receptionist, Genetics Clinic"),
    ("pharmacy@medipurse.test", "Pharmacist, Genetics Clinic"),
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


def seed_prescriptions_if_empty() -> None:
    """Demo medicines (also added to databases created before medicines existed)."""
    db = SessionLocal()
    try:
        if db.query(Prescription).count():
            return
        mehta = db.query(User).filter(User.email == "dr.mehta@medipurse.test").first()
        rao = db.query(User).filter(User.email == "dr.rao@medipurse.test").first()
        if not mehta or not rao:
            return

        def pt(first, last):
            return db.query(Patient).filter(Patient.first_name == first, Patient.last_name == last).first()

        def dx(p, cond):
            return next((d for d in p.diagnoses if d.condition == cond), None) if p else None

        now = datetime.now().replace(second=0, microsecond=0)
        rows = [
            (pt("Sunita", "Verma"), mehta, "Pseudoxanthoma elasticum", now - timedelta(hours=2),
             "Vitamin E", "400 IU", "Once daily", "3 months", "After breakfast"),
            (pt("Sunita", "Verma"), mehta, "Pseudoxanthoma elasticum", now - timedelta(hours=2),
             "Carboxymethylcellulose eye drops", "1 drop", "Four times daily", "1 month", "Both eyes"),
            (pt("Kavya", "Nair"), mehta, "Hypermobile Ehlers-Danlos syndrome", now - timedelta(days=1, hours=3),
             "Paracetamol", "500 mg", "As needed, max 3/day", "5 days", "For joint pain"),
            (pt("Rohan", "Gupta"), rao, "Essential tremor", now - timedelta(days=1, hours=5),
             "Propranolol", "40 mg", "Twice daily", "1 month", "Check pulse before each dose"),
            (pt("Rohan", "Gupta"), rao, "Essential tremor", now - timedelta(days=30),
             "Propranolol", "20 mg", "Twice daily", "1 month", "Starting dose"),
        ]
        for p, doc, cond, when, med, dose, freq, dur, instr in rows:
            if not p:
                continue
            d = dx(p, cond)
            db.add(Prescription(patient_id=p.id, doctor_id=doc.id, diagnosis_id=d.id if d else None,
                                medicine=med, dosage=dose, frequency=freq, duration=dur,
                                instructions=instr, prescribed_at=when,
                                status="stopped" if when < now - timedelta(days=20) else "active",
                                stopped_on=(when + timedelta(days=28)).date() if when < now - timedelta(days=20) else None))
        db.commit()
    finally:
        db.close()


DEMO_PATIENT_EMAIL = "sunita.verma@example.test"


def seed_portal_demo() -> None:
    """Give one demo patient an email and a portal password (local testing only)."""
    db = SessionLocal()
    try:
        p = db.query(Patient).filter(Patient.first_name == "Sunita", Patient.last_name == "Verma").first()
        if p and not p.password_hash and not p.email:
            p.email = DEMO_PATIENT_EMAIL
            p.password_hash = hash_password(DEMO_PASSWORD)
            db.commit()
    finally:
        db.close()


def seed_inventory_if_empty() -> None:
    """Demo pharmacy stock for every clinic (some low, out of stock or expiring, to show alerts)."""
    from .db import Drug, StockMove
    from .pharmacy import COMMON_MEDS, FORM_UNIT
    db = SessionLocal()
    try:
        if db.query(Drug).count():
            return
        special = {"Montelukast": 0, "Salbutamol": 3, "Azithromycin": 12, "Vitamin D3": 8}
        for c in db.query(Clinic).all():
            for i, (name, strength, form, *_rest) in enumerate(COMMON_MEDS):
                stock = special.get(name, 60 + (i * 37) % 240)
                if form not in ("tablet", "capsule", "sachet"):
                    stock = special.get(name, 6 + i % 10)
                expiry = date.today() + timedelta(days=40 if name == "Cetirizine" else 400 + i * 15)
                d = Drug(clinic_id=c.id, name=name, strength=strength, form=form, unit=FORM_UNIT.get(form, "unit"),
                         stock=stock, reorder_level=5 if form not in ("tablet", "capsule", "sachet") else 30,
                         batch=f"B{2400 + i}", expiry=expiry)
                db.add(d)
                db.flush()
                db.add(StockMove(drug_id=d.id, change=stock, balance=stock, reason="added", note="Opening stock"))
        db.commit()
    finally:
        db.close()


def seed_pharmacy_extras() -> None:
    """Demo pharmacist (Genetics Clinic) and unit prices; safe to run on older databases."""
    from .db import Drug
    from .pharmacy import PRICES
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "pharmacy@medipurse.test").first():
            gen = db.query(Clinic).filter(Clinic.name == "Genetics Clinic").first()
            if gen:
                db.add(User(name="Rahul Desai", email="pharmacy@medipurse.test", role="pharmacist",
                            clinic_id=gen.id, password_hash=hash_password(DEMO_PASSWORD)))
        for d in db.query(Drug).filter((Drug.price == None) | (Drug.price == 0)).all():  # noqa: E711
            if d.name in PRICES:
                d.price = PRICES[d.name]
        db.commit()
    finally:
        db.close()
