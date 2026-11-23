"""Medicines suggested for a diagnosis on the prescription form.

  - Everyday conditions (the realist's 51 COMMON_CONDITIONS): usual first-line options in
    adults, written for this app. Prompts for the doctor, not orders: doses are typical
    starting points to check and adjust.
  - Rare diseases (Orphanet id on the diagnosis): medicines actually used in published
    ZebraMap case reports for that disease, with how many cases used each, plus approved
    indications and contraindications from OptimusKG (app/data/disease_drugs.json, built by
    scripts/build_drug_suggestions.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# (name, strength, form, dose, frequency, duration, instructions)
COMMON_CONDITION_MEDS = {
    "Viral upper respiratory infection (common cold)": [("Paracetamol", "500 mg", "tablet", "1 tablet", "Three times daily", "3 days", "After food"), ("Cetirizine", "10 mg", "tablet", "1 tablet", "At bedtime", "5 days", "")],
    "Influenza-like illness": [("Paracetamol", "500 mg", "tablet", "1 tablet", "Three times daily", "5 days", "After food"), ("Oseltamivir", "75 mg", "capsule", "1 capsule", "Twice daily", "5 days", "Start within 48 hours of symptoms")],
    "Acute pharyngitis / tonsillitis": [("Phenoxymethylpenicillin", "500 mg", "tablet", "1 tablet", "Four times daily", "10 days", "Complete the full course"), ("Paracetamol", "500 mg", "tablet", "1 tablet", "Three times daily", "3 days", "After food")],
    "Acute sinusitis": [("Mometasone nasal spray", "50 mcg", "other", "2 puffs", "Once daily", "14 days", ""), ("Amoxicillin", "500 mg", "capsule", "1 capsule", "Three times daily", "5 days", "Only if symptoms worsen after 10 days")],
    "Acute bronchitis": [("Paracetamol", "500 mg", "tablet", "1 tablet", "Three times daily", "3 days", "After food"), ("Dextromethorphan syrup", "10 mg/5 ml", "syrup", "10 ml", "Three times daily", "5 days", "Shake well before use")],
    "Community-acquired pneumonia": [("Amoxicillin", "500 mg", "capsule", "1 capsule", "Three times daily", "5 days", "Complete the full course"), ("Azithromycin", "500 mg", "tablet", "1 tablet", "Once daily", "3 days", "If atypical infection suspected")],
    "Asthma": [("Salbutamol", "100 mcg", "inhaler", "2 puffs", "As needed", "1 month", "Rinse mouth after use"), ("Budesonide-formoterol", "200/6 mcg", "inhaler", "1 puff", "Twice daily", "1 month", "Rinse mouth after use"), ("Montelukast", "10 mg", "tablet", "1 tablet", "At bedtime", "1 month", "")],
    "Allergic rhinitis": [("Cetirizine", "10 mg", "tablet", "1 tablet", "Once daily", "14 days", ""), ("Fluticasone nasal spray", "50 mcg", "other", "2 puffs", "Once daily", "1 month", "")],
    "Allergic conjunctivitis / viral conjunctivitis": [("Olopatadine eye drops", "0.1%", "drops", "1 drop", "Twice daily", "14 days", "Each eye"), ("Carboxymethylcellulose eye drops", "0.5%", "drops", "1 drop", "Four times daily", "7 days", "")],
    "Acute otitis media / otitis externa": [("Amoxicillin", "500 mg", "capsule", "1 capsule", "Three times daily", "5 days", "If no better after 48 hours"), ("Ciprofloxacin ear drops", "0.3%", "drops", "2 drops", "Twice daily", "7 days", "For otitis externa")],
    "Acute gastroenteritis": [("ORS", "21 g", "sachet", "1 sachet", "As needed", "3 days", "Dissolve in 1 litre of water"), ("Ondansetron", "4 mg", "tablet", "1 tablet", "As needed", "3 days", "")],
    "Gastro-oesophageal reflux disease": [("Pantoprazole", "40 mg", "tablet", "1 tablet", "Once daily", "1 month", "Before breakfast")],
    "Dyspepsia / gastritis / peptic ulcer": [("Pantoprazole", "40 mg", "tablet", "1 tablet", "Once daily", "1 month", "Before breakfast"), ("Domperidone", "10 mg", "tablet", "1 tablet", "Three times daily", "7 days", "Before food")],
    "Irritable bowel syndrome": [("Mebeverine", "135 mg", "tablet", "1 tablet", "Three times daily", "1 month", "Before food"), ("Ispaghula husk", "3.5 g", "sachet", "1 sachet", "Twice daily", "1 month", "With plenty of water")],
    "Functional constipation": [("Macrogol", "13.8 g", "sachet", "1 sachet", "Once daily", "14 days", "With plenty of water"), ("Ispaghula husk", "3.5 g", "sachet", "1 sachet", "Twice daily", "1 month", "With plenty of water")],
    "Haemorrhoids / anal fissure": [("Lactulose", "10 g/15 ml", "syrup", "15 ml", "Twice daily", "14 days", ""), ("Glyceryl trinitrate ointment", "0.4%", "ointment", "Apply thin layer", "Twice daily", "8 weeks", "Apply on affected area")],
    "Urinary tract infection": [("Nitrofurantoin", "100 mg", "capsule", "1 capsule", "Twice daily", "3 days", "With food")],
    "Acute pyelonephritis": [("Ciprofloxacin", "500 mg", "tablet", "1 tablet", "Twice daily", "7 days", "Complete the full course")],
    "Kidney stone (renal colic)": [("Diclofenac", "50 mg", "tablet", "1 tablet", "Three times daily", "5 days", "After food"), ("Tamsulosin", "400 mcg", "capsule", "1 capsule", "Once daily", "14 days", "")],
    "Type 2 diabetes mellitus": [("Metformin", "500 mg", "tablet", "1 tablet", "Twice daily", "1 month", "After food")],
    "Hypothyroidism": [("Levothyroxine", "50 mcg", "tablet", "1 tablet", "In the morning", "1 month", "Empty stomach")],
    "Hyperthyroidism": [("Carbimazole", "20 mg", "tablet", "1 tablet", "Once daily", "1 month", ""), ("Propranolol", "40 mg", "tablet", "1 tablet", "Twice daily", "1 month", "")],
    "Iron-deficiency anaemia": [("Ferrous sulfate", "200 mg", "tablet", "1 tablet", "Once daily", "3 months", "Empty stomach")],
    "Essential hypertension": [("Amlodipine", "5 mg", "tablet", "1 tablet", "Once daily", "1 month", ""), ("Losartan", "50 mg", "tablet", "1 tablet", "Once daily", "1 month", "")],
    "Tension-type headache": [("Paracetamol", "500 mg", "tablet", "2 tablets", "As needed", "5 days", "After food"), ("Ibuprofen", "400 mg", "tablet", "1 tablet", "As needed", "5 days", "After food")],
    "Migraine": [("Sumatriptan", "50 mg", "tablet", "1 tablet", "As needed", "1 month", "At the start of the headache"), ("Ibuprofen", "400 mg", "tablet", "1 tablet", "As needed", "1 month", "After food")],
    "Benign paroxysmal positional vertigo": [("Prochlorperazine", "5 mg", "tablet", "1 tablet", "As needed", "3 days", "Short term only")],
    "Mechanical low back pain": [("Ibuprofen", "400 mg", "tablet", "1 tablet", "Three times daily", "5 days", "After food"), ("Paracetamol", "500 mg", "tablet", "2 tablets", "Three times daily", "5 days", "")],
    "Cervical spondylosis / neck strain": [("Ibuprofen", "400 mg", "tablet", "1 tablet", "Three times daily", "5 days", "After food")],
    "Knee osteoarthritis": [("Paracetamol", "500 mg", "tablet", "2 tablets", "Three times daily", "1 month", ""), ("Diclofenac gel", "1%", "gel", "Apply thin layer", "Three times daily", "14 days", "Apply on affected area")],
    "Rheumatoid arthritis": [("Methotrexate", "2.5 mg", "tablet", "3 tablets", "Once weekly", "1 month", "Same day each week; take folic acid"), ("Folic acid", "5 mg", "tablet", "1 tablet", "Once weekly", "1 month", "Not on the methotrexate day")],
    "Gout": [("Colchicine", "500 mcg", "tablet", "1 tablet", "Twice daily", "5 days", ""), ("Naproxen", "500 mg", "tablet", "1 tablet", "Twice daily", "5 days", "After food")],
    "Frozen shoulder / rotator cuff disorder": [("Ibuprofen", "400 mg", "tablet", "1 tablet", "Three times daily", "7 days", "After food")],
    "Vitamin D or B12 deficiency": [("Vitamin D3", "60,000 IU", "capsule", "1 capsule", "Once weekly", "8 weeks", "With food"), ("Cyanocobalamin", "1,000 mcg", "tablet", "1 tablet", "Once daily", "1 month", "")],
    "Atopic dermatitis (eczema)": [("Emollient cream", "", "cream", "Apply thin layer", "Three times daily", "1 month", "Apply on affected area"), ("Hydrocortisone cream", "1%", "cream", "Apply thin layer", "Twice daily", "7 days", "Apply on affected area")],
    "Urticaria / allergic reaction": [("Cetirizine", "10 mg", "tablet", "1 tablet", "Once daily", "7 days", "")],
    "Fungal skin infection": [("Clotrimazole cream", "1%", "cream", "Apply thin layer", "Twice daily", "14 days", "Apply on affected area")],
    "Cellulitis": [("Flucloxacillin", "500 mg", "capsule", "1 capsule", "Four times daily", "7 days", "Empty stomach")],
    "Acne vulgaris": [("Adapalene-benzoyl peroxide gel", "0.1%/2.5%", "gel", "Apply thin layer", "At bedtime", "3 months", "Apply on affected area")],
    "Viral exanthem": [("Paracetamol", "500 mg", "tablet", "1 tablet", "As needed", "3 days", "")],
    "Dengue / other acute febrile illness (endemic areas)": [("Paracetamol", "500 mg", "tablet", "1 tablet", "Four times daily", "5 days", "Avoid ibuprofen and aspirin"), ("ORS", "21 g", "sachet", "1 sachet", "As needed", "5 days", "Dissolve in 1 litre of water")],
    "Generalised anxiety disorder": [("Sertraline", "50 mg", "tablet", "1 tablet", "Once daily", "1 month", "")],
    "Depression": [("Sertraline", "50 mg", "tablet", "1 tablet", "Once daily", "1 month", "")],
    "Insomnia": [("Melatonin", "2 mg", "tablet", "1 tablet", "At bedtime", "14 days", "")],
    "Dysmenorrhoea": [("Mefenamic acid", "500 mg", "tablet", "1 tablet", "Three times daily", "3 days", "With food")],
    "Polycystic ovary syndrome": [("Metformin", "500 mg", "tablet", "1 tablet", "Twice daily", "1 month", "After food")],
    "Vaginal infection (candida, bacterial vaginosis)": [("Fluconazole", "150 mg", "capsule", "1 capsule", "Single dose", "1 day", ""), ("Metronidazole", "400 mg", "tablet", "1 tablet", "Twice daily", "7 days", "Avoid alcohol")],
    "Dental infection / caries": [("Amoxicillin", "500 mg", "capsule", "1 capsule", "Three times daily", "5 days", ""), ("Metronidazole", "400 mg", "tablet", "1 tablet", "Three times daily", "5 days", "Avoid alcohol")],
    "Aphthous ulcers": [("Chlorhexidine mouthwash", "0.2%", "other", "10 ml", "Twice daily", "7 days", "Rinse and spit out")],
    "Stable angina / coronary artery disease": [("Aspirin", "75 mg", "tablet", "1 tablet", "Once daily", "1 month", "After food"), ("Glyceryl trinitrate spray", "400 mcg", "other", "1 puff", "As needed", "1 month", "Under the tongue")],
    "Heart failure": [("Furosemide", "40 mg", "tablet", "1 tablet", "In the morning", "1 month", ""), ("Bisoprolol", "1.25 mg", "tablet", "1 tablet", "Once daily", "1 month", "")],
}

_DISEASE_DRUGS: dict | None = None


def disease_drugs() -> dict:
    global _DISEASE_DRUGS
    if _DISEASE_DRUGS is None:
        f = Path(__file__).parent / "data" / "disease_drugs.json"
        _DISEASE_DRUGS = json.loads(f.read_text(encoding="utf-8"))["diseases"] if f.exists() else {}
    return _DISEASE_DRUGS


def _common_match(condition: str) -> str | None:
    c = condition.strip().lower()
    if not c:
        return None
    for name in COMMON_CONDITION_MEDS:
        n = name.lower()
        core = re.split(r" \(| / ", n)[0]
        if c == n or c == core or (len(c) > 4 and (c in n or core in c)):
            return name
    return None


def suggestions_for(dx) -> dict:
    """Suggestions for one Diagnosis row."""
    out = {"source": "", "items": [], "contraindicated": []}
    name = _common_match(dx.condition or "")
    if name:
        out["source"] = "common"
        for m in COMMON_CONDITION_MEDS[name]:
            out["items"].append({"label": " ".join(x for x in m[:3] if x), "form": m[2], "dose": m[3],
                                 "frequency": m[4], "duration": m[5], "instructions": m[6],
                                 "why": "usual first-line option"})
        return out
    rec = disease_drugs().get(str(dx.disease_id or ""))
    if rec:
        out["source"] = "rare"
        ind = set(rec.get("indicated") or [])
        for d in rec.get("used") or []:
            out["items"].append({"label": d["name"], "cases": d["cases"],
                                 "why": f"used in {d['cases']} of {rec['treated_cases']} published cases"
                                        + (" · approved indication" if d["name"] in ind else "")})
        seen = {i["label"] for i in out["items"]}
        for nm in rec.get("indicated") or []:
            if nm not in seen:
                out["items"].append({"label": nm, "why": "approved indication"})
        out["contraindicated"] = rec.get("contraindicated") or []
        out["treated_cases"] = rec.get("treated_cases")
    return out


BRANDS = {"pcm": "paracetamol", "crocin": "paracetamol", "dolo": "paracetamol", "calpol": "paracetamol",
          "acetaminophen": "paracetamol", "tylenol": "paracetamol", "brufen": "ibuprofen", "combiflam": "ibuprofen",
          "augmentin": "amoxicillin", "mox": "amoxicillin", "azee": "azithromycin", "zithromax": "azithromycin",
          "azithral": "azithromycin", "cetzine": "cetirizine", "okacet": "cetirizine", "pan": "pantoprazole",
          "pantop": "pantoprazole", "omez": "omeprazole", "rantac": "ranitidine", "zofer": "ondansetron",
          "emeset": "ondansetron", "domstal": "domperidone", "glycomet": "metformin", "amlong": "amlodipine",
          "montair": "montelukast", "asthalin": "salbutamol", "albuterol": "salbutamol", "voveran": "diclofenac",
          "disprin": "aspirin", "ecosprin": "aspirin", "flagyl": "metronidazole", "cipro": "ciprofloxacin",
          "ciplox": "ciprofloxacin", "septran": "co-trimoxazole", "bactrim": "co-trimoxazole"}


def generic(name: str) -> str:
    """'Paracetamol 500 mg tablet' -> 'paracetamol' (for matching reactions and contraindications)."""
    words = re.findall(r"[a-z][a-z-]+", (name or "").lower())
    stop = {"tablet", "tablets", "capsule", "capsules", "syrup", "cream", "gel", "drops", "eye", "ear", "nasal",
            "spray", "inhaler", "ointment", "sachet", "injection", "mg", "mcg", "ml", "iu", "suspension", "oral"}
    keep = [BRANDS.get(w, w) for w in words if w not in stop]
    return keep[0] if keep else (name or "").lower().strip()


def same_medicine(a: str, b: str) -> bool:
    """True if two medicine names refer to the same drug (by generic name)."""
    ga, gb = generic(a), generic(b)
    return bool(ga) and ga == gb
