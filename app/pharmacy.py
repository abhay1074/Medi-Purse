"""Clinic pharmacy: drug stock, quick-pick prescribing, dispensing and counter sales.

Who does what:
  doctor        prescribes (sees stock, never changes it)
  pharmacist    manages stock, dispenses prescriptions, sells over the counter
  receptionist  dispenses prescriptions; manages stock only if the clinic has no pharmacist
  admin         everything, all clinics

When a doctor prescribes, each medicine is matched to the clinic's stock and the units
for the course are worked out (dose x frequency x duration). Nothing is deducted yet:
the prescription waits in the dispensing queue. The pharmacist or receptionist hands
over what the clinic has (stock goes down then); the rest is marked "buy from another
pharmacy".
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta

from .db import Drug, StockMove

FORMS = [("tablet", "tablet"), ("capsule", "capsule"), ("syrup", "bottle"), ("suspension", "bottle"),
         ("injection", "vial"), ("drops", "bottle"), ("inhaler", "inhaler"), ("cream", "tube"),
         ("ointment", "tube"), ("gel", "tube"), ("sachet", "sachet"), ("other", "unit")]
FORM_UNIT = dict(FORMS)
COUNTED = {"tablet", "capsule", "sachet"}   # forms counted per dose; others are dispensed per pack

# dropdowns on the prescription form
DOSES = ["1 tablet", "½ tablet", "2 tablets", "1 capsule", "2 capsules", "5 ml", "10 ml", "15 ml",
         "1 sachet", "1 puff", "2 puffs", "1 drop", "2 drops", "1 injection", "Apply thin layer"]
FREQUENCIES = [("Once daily", 1), ("Twice daily", 2), ("Three times daily", 3), ("Four times daily", 4),
               ("At bedtime", 1), ("In the morning", 1), ("Every 6 hours", 4), ("Every 8 hours", 3),
               ("As needed", 0), ("Once weekly", 1 / 7), ("Single dose", -1)]
FREQ_PER_DAY = dict(FREQUENCIES)
DURATIONS = [("1 day", 1), ("3 days", 3), ("5 days", 5), ("7 days", 7), ("10 days", 10), ("14 days", 14),
             ("1 month", 30), ("2 months", 60), ("3 months", 90), ("8 weeks", 56), ("Ongoing", 30)]
DURATION_DAYS = dict(DURATIONS)
INSTRUCTIONS = ["After food", "Before food", "With food", "Empty stomach", "Before breakfast",
                "At bedtime", "With plenty of water", "Do not crush or chew", "Shake well before use",
                "Complete the full course", "Avoid alcohol", "Apply on affected area", "Rinse mouth after use"]
AS_NEEDED_DAYS_DOSES = 2   # "as needed": assume up to 2 doses a day for the course

# common medicines with their usual adult course (name, strength, form, dose, frequency, duration, instructions)
COMMON_MEDS = [
    ("Paracetamol", "500 mg", "tablet", "1 tablet", "Three times daily", "3 days", "After food"),
    ("Ibuprofen", "400 mg", "tablet", "1 tablet", "Three times daily", "3 days", "After food"),
    ("Cetirizine", "10 mg", "tablet", "1 tablet", "At bedtime", "5 days", ""),
    ("Amoxicillin", "500 mg", "capsule", "1 capsule", "Three times daily", "5 days", "Complete the full course"),
    ("Azithromycin", "500 mg", "tablet", "1 tablet", "Once daily", "3 days", "Before food"),
    ("Pantoprazole", "40 mg", "tablet", "1 tablet", "Once daily", "14 days", "Before breakfast"),
    ("Ondansetron", "4 mg", "tablet", "1 tablet", "As needed", "3 days", ""),
    ("Domperidone", "10 mg", "tablet", "1 tablet", "Three times daily", "3 days", "Before food"),
    ("ORS", "21 g", "sachet", "1 sachet", "As needed", "3 days", "Dissolve in 1 litre of water"),
    ("Metformin", "500 mg", "tablet", "1 tablet", "Twice daily", "1 month", "After food"),
    ("Amlodipine", "5 mg", "tablet", "1 tablet", "Once daily", "1 month", ""),
    ("Montelukast", "10 mg", "tablet", "1 tablet", "At bedtime", "14 days", ""),
    ("Salbutamol", "100 mcg", "inhaler", "2 puffs", "As needed", "1 month", "Rinse mouth after use"),
    ("Dextromethorphan syrup", "10 mg/5 ml", "syrup", "10 ml", "Three times daily", "5 days", "Shake well before use"),
    ("Vitamin D3", "60,000 IU", "capsule", "1 capsule", "Once weekly", "8 weeks", "With food"),
    ("Folic acid", "5 mg", "tablet", "1 tablet", "Once daily", "1 month", ""),
]


def _dose_units(dose: str) -> float:
    d = (dose or "").strip().lower()
    if d.startswith("½") or d.startswith("1/2"):
        return 0.5
    m = re.match(r"(\d+(?:\.\d+)?)", d)
    return float(m.group(1)) if m else 1.0


def units_needed(form: str, dose: str, frequency: str, duration: str) -> int:
    """Units of stock for the whole course. Tablets/capsules/sachets are counted per dose;
    syrups, inhalers, creams and similar are one pack."""
    form = (form or "tablet").lower()
    if form not in COUNTED:
        return 1
    per_day = FREQ_PER_DAY.get(frequency)
    days = DURATION_DAYS.get(duration)
    if days is None:
        m = re.match(r"(\d+)\s*(day|week|month)", (duration or "").lower())
        days = int(m.group(1)) * {"day": 1, "week": 7, "month": 30}[m.group(2)] if m else 0
    dose_n = _dose_units(dose)
    if per_day == -1:
        return max(1, math.ceil(dose_n))
    if per_day == 0:   # as needed
        per_day = AS_NEEDED_DAYS_DOSES
    if per_day is None or not days:
        return 0
    return max(1, math.ceil(dose_n * per_day * days))


def clinic_drugs(db, clinic_id: int | None, active_only=True):
    q = db.query(Drug)
    if clinic_id:
        q = q.filter(Drug.clinic_id == clinic_id)
    if active_only:
        q = q.filter(Drug.active == 1)
    return q.order_by(Drug.name, Drug.strength).all()


def find_drug(db, clinic_id: int, text: str, drug_id: int | None = None) -> Drug | None:
    if drug_id:
        d = db.get(Drug, drug_id)
        if d and d.clinic_id == clinic_id and d.active:
            return d
    t = " ".join((text or "").lower().split())
    if not t:
        return None
    best = None
    for d in clinic_drugs(db, clinic_id):
        if d.label.lower() == t or f"{d.name} {d.strength}".lower().strip() == t:
            return d
        if d.name.lower() == t and best is None:
            best = d
    return best


def move(db, drug: Drug, change: int, reason: str, user_id: int | None = None, note: str = "",
         prescription_id: int | None = None, sale_id: int | None = None) -> StockMove:
    drug.stock = max(0, drug.stock + change)
    m = StockMove(drug_id=drug.id, change=change, balance=drug.stock, reason=reason, note=note[:300],
                  user_id=user_id, prescription_id=prescription_id, sale_id=sale_id, created_at=datetime.now())
    db.add(m)
    return m


# ----------------------------------------------------------------------------
# roles
# ----------------------------------------------------------------------------
DISPENSERS = ("pharmacist", "receptionist", "admin")
SELLERS = ("pharmacist", "admin")


def clinic_has_pharmacist(db, clinic_id: int | None) -> bool:
    from .db import User
    return bool(clinic_id) and db.query(User).filter(User.clinic_id == clinic_id, User.role == "pharmacist",
                                                     User.active == 1).count() > 0


def can_manage_stock(db, user) -> bool:
    if user.role in ("admin", "pharmacist"):
        return True
    return user.role == "receptionist" and not clinic_has_pharmacist(db, user.clinic_id)


def can_dispense(user) -> bool:
    return user.role in DISPENSERS


def can_sell(user) -> bool:
    return user.role in SELLERS


def usable(drug: Drug | None) -> int:
    """Units that can be handed out (expired stock never is)."""
    if drug is None or not drug.active or (drug.expiry and drug.expiry < date.today()):
        return 0
    return drug.stock


# ----------------------------------------------------------------------------
# prescriptions: prescribe -> queue -> dispense
# ----------------------------------------------------------------------------
def prepare(rx, drug: Drug | None, quantity: int) -> None:
    """At prescribing time: link to stock and set the course quantity; no stock change."""
    rx.drug_id = drug.id if drug else None
    q = max(0, int(quantity or 0))
    if not q and drug is not None and drug.form not in COUNTED:
        q = 1
    rx.quantity = q
    rx.dispensed, rx.supply, rx.dispense_status = 0, "", "pending"


def availability(rx) -> str:
    """What the clinic can supply right now (shown before dispensing)."""
    have = usable(rx.drug)
    if rx.drug is None or have <= 0:
        return "none"
    return "full" if not rx.quantity or have >= rx.quantity else "some"


def dispense_line(db, rx, drug: Drug | None, give: int, user_id: int | None) -> None:
    """Hand over `give` units from clinic stock (capped at what is usable) and close the line."""
    if drug is not None:
        rx.drug_id = drug.id
    give = max(0, min(int(give or 0), usable(drug)))
    want = rx.quantity or give
    rx.quantity = want
    rx.dispensed = give
    rx.supply = "clinic" if give and give >= want else ("partial" if give else "outside")
    rx.dispense_status = "done"
    rx.dispensed_by_id = user_id
    rx.dispensed_at = datetime.now()
    if give:
        move(db, drug, -give, "dispensed", user_id, note=f"Prescription for patient #{rx.patient_id}",
             prescription_id=rx.id)


def pending_groups(db, clinic_id: int | None, status: str = "pending", limit: int = 200) -> list[dict]:
    """Prescriptions grouped as the doctor saved them (one group = one visit's medicines)."""
    from .db import Patient, Prescription
    q = db.query(Prescription).join(Patient).filter(Prescription.dispense_status == status)
    if clinic_id:
        q = q.filter(Patient.clinic_id == clinic_id)
    order = Prescription.prescribed_at.asc() if status == "pending" else Prescription.dispensed_at.desc()
    groups: dict[str, dict] = {}
    for rx in q.order_by(order).limit(limit).all():
        key = rx.rx_group or f"rx{rx.id}"
        g = groups.setdefault(key, {"key": key, "patient": rx.patient, "doctor": rx.doctor, "lines": [],
                                    "at": rx.prescribed_at, "done_at": rx.dispensed_at, "by": rx.dispensed_by})
        g["lines"].append(rx)
    return list(groups.values())


def group_lines(db, key: str):
    from .db import Prescription
    if key.startswith("rx") and key[2:].isdigit():
        rx = db.get(Prescription, int(key[2:]))
        return [rx] if rx else []
    return db.query(Prescription).filter(Prescription.rx_group == key).order_by(Prescription.id).all()


# ----------------------------------------------------------------------------
# counter sales
# ----------------------------------------------------------------------------
PAYMENTS = [("cash", "Cash"), ("upi", "UPI"), ("card", "Card")]
PRICES = {"Paracetamol": 1.5, "Ibuprofen": 2.0, "Cetirizine": 2.5, "Amoxicillin": 6.0, "Azithromycin": 22.0,
          "Pantoprazole": 7.0, "Ondansetron": 5.0, "Domperidone": 3.0, "ORS": 20.0, "Metformin": 2.0,
          "Amlodipine": 3.0, "Montelukast": 12.0, "Salbutamol": 160.0, "Dextromethorphan syrup": 95.0,
          "Vitamin D3": 30.0, "Folic acid": 1.5, "Doxycycline": 8.0}


def sell(db, sale, lines: list[tuple], user_id: int | None) -> list[str]:
    """lines: (drug, qty, unit_price). Sells only what is usable; returns problems found."""
    from .db import SaleItem
    problems, total = [], 0.0
    db.add(sale)
    db.flush()
    for drug, qty, price in lines:
        qty = int(qty or 0)
        if qty <= 0:
            continue
        if qty > usable(drug):
            problems.append(f"{drug.label}: only {usable(drug)} available")
            continue
        price = float(price if price is not None else drug.price or 0)
        amount = round(price * qty, 2)
        db.add(SaleItem(sale_id=sale.id, drug_id=drug.id, name=drug.label, quantity=qty, unit_price=price,
                        amount=amount))
        move(db, drug, -qty, "sold", user_id, sale_id=sale.id,
             note=f"Counter sale #{sale.id}" + (f" to {sale.customer_name}" if sale.customer_name else ""))
        total += amount
    sale.total = round(total, 2)
    return problems


def supply_note(rx) -> str:
    """Plain wording for patients and staff."""
    if getattr(rx, "dispense_status", "") == "pending":
        return "Waiting at the clinic pharmacy"
    if rx.supply == "clinic":
        return "Given by the clinic pharmacy"
    if rx.supply == "partial":
        rest = max(0, (rx.quantity or 0) - (rx.dispensed or 0))
        return f"{rx.dispensed} given by the clinic; buy the remaining {rest} from another pharmacy"
    if rx.supply == "outside":
        return "Not available at the clinic: buy from another pharmacy"
    return ""


def low_stock(db, clinic_id: int | None) -> list[Drug]:
    return [d for d in clinic_drugs(db, clinic_id) if d.level != "ok"]


def expiring(db, clinic_id: int | None, days: int = 60) -> list[Drug]:
    limit = date.today() + timedelta(days=days)
    return [d for d in clinic_drugs(db, clinic_id) if d.expiry and d.expiry <= limit and d.stock > 0]


def form_options(db, clinic_id: int | None) -> dict:
    """Everything the fast prescription form needs, as JSON-friendly data."""
    drugs = clinic_drugs(db, clinic_id)
    stock = [{"id": d.id, "label": d.label, "name": d.name, "strength": d.strength, "form": d.form,
              "unit": d.unit, "stock": d.stock if not (d.expiry and d.expiry < date.today()) else 0,
              "expired": bool(d.expiry and d.expiry < date.today())} for d in drugs]
    common = []
    for name, strength, form, dose, freq, dur, instr in COMMON_MEDS:
        label = " ".join(x for x in (name, strength, form) if x)
        d = next((x for x in stock if x["label"].lower() == label.lower()), None)
        common.append({"label": label, "form": form, "dose": dose, "frequency": freq, "duration": dur,
                       "instructions": instr, "drug_id": d["id"] if d else None, "stock": d["stock"] if d else 0})
    return {"stock": stock, "common": common, "prices": {d.id: d.price for d in drugs}, "doses": DOSES, "frequencies": [f for f, _ in FREQUENCIES],
            "durations": [d for d, _ in DURATIONS], "instructions": INSTRUCTIONS,
            "per_day": {f: n for f, n in FREQUENCIES}, "days": {d: n for d, n in DURATIONS},
            "counted": sorted(COUNTED), "as_needed": AS_NEEDED_DAYS_DOSES}
