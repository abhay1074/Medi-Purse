"""Symptom and phenotype search over the full OptimusKG phenotype list plus the
curated everyday-symptom list (app/clinical.py).

Used by:
  /api/symptoms           type-ahead search for doctors and patients
  realist agent           find which phenotypes a free-text case mentions
  history-taking agent    turn the patient's answers into clinical terms
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from .clinical import COMMON_BY_KEY, COMMON_SYMPTOMS

DATA = Path(__file__).parent / "data" / "phenotypes.json"
_WORD = re.compile(r"[a-z0-9]+")


def norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def stem_phrase(s: str) -> str:
    return " ".join(_stem(w) for w in norm(s).split())


def _variants(phrase: str) -> set[str]:
    """Matching variants: stemmed, without '(...)', without a trailing 'of the X'."""
    base = re.sub(r"\([^)]*\)", " ", phrase)
    out = {stem_phrase(phrase), stem_phrase(base)}
    m = re.match(r"(.+?) (?:of|in|on) the [a-z ]+$", norm(base))
    if m and len(m.group(1).split()) >= 2:
        out.add(stem_phrase(m.group(1)))
    return {v for v in out if v}


# everyday words patients use -> clinical term (resolved through the index)
COLLOQUIAL = {
    "dizzy": "dizziness", "light headed": "dizziness", "lightheaded": "dizziness", "giddiness": "dizziness",
    "giddy": "dizziness", "throwing up": "vomiting", "vomit": "vomiting", "tired": "fatigue",
    "exhausted": "fatigue", "short of breath": "dyspnea", "breathless": "dyspnea", "breathlessness": "dyspnea",
    "tummy ache": "abdominal pain", "stomach ache": "abdominal pain", "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain", "loose motion": "diarrhea", "loose stool": "diarrhea",
    "acidity": "heartburn", "gas": "abdominal bloating", "temperature": "fever",
    "high temperature": "fever", "feverish": "fever", "itchy": "pruritus",
    "itchiness": "pruritus", "pins and needle": "paresthesia", "tingling": "paresthesia", "numb": "paresthesia",
    "stuffy nose": "nasal congestion", "throat pain": "sore throat", "painful throat": "sore throat",
    "cant sleep": "insomnia", "cannot sleep": "insomnia", "sleepless": "insomnia", "sad": "depressed mood",
    "worried": "anxiety", "heart racing": "palpitations", "racing heart": "palpitations",
    "passed out": "syncope", "blacked out": "syncope", "fainted": "syncope", "swollen feet": "pedal edema",
    "swollen ankle": "pedal edema", "burning urine": "dysuria", "peeing a lot": "polyuria",
    "back ache": "low back pain", "backache": "low back pain", "joint ache": "arthralgia",
    "aching joints": "arthralgia", "muscle ache": "myalgia", "body ache": "myalgia", "body pain": "myalgia",
    "losing weight": "weight loss", "yellow eyes": "jaundice", "yellow skin": "jaundice",
    "pimples": "acne", "hair fall": "hair loss", "rash": "skin rash", "fits": "seizure",
    "forgetful": "memory impairment", "blurry": "blurred vision", "blurriness": "blurred vision",
    "vision is blurred": "blurred vision", "cant see clearly": "blurred vision", "double vision": "diplopia",
    "seeing double": "diplopia", "ringing in my ear": "tinnitus", "out of breath": "dyspnea",
    "chest hurt": "chest pain", "head hurt": "headache", "head ache": "headache", "sore eyes": "eye pain",
    "peeing often": "urinary frequency", "burning pee": "dysuria", "throat is sore": "sore throat",
    "coughing": "cough", "sneezy": "sneezing", "shaky": "tremor", "trembling": "tremor",
}


class SymptomIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self.items: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.phrase: dict[str, dict] = {}   # normalised name/synonym -> item
        self.max_words = 1
        self.source = ""
        self._search: list[tuple] = []

    def load(self) -> "SymptomIndex":
        with self._lock:
            if self._loaded:
                return self
            payload = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {"items": []}
            self.source = payload.get("source", "")
            common_by_hpo = {s[4]: s for s in COMMON_SYMPTOMS if s[4]}
            for pid, name, syn, cat, n in payload["items"]:
                item = {"id": pid, "name": name, "synonyms": syn, "category": cat, "diseases": n,
                        "common": pid in common_by_hpo}
                self.items.append(item)
                self.by_id[pid] = item
                # precomputed search fields
                syns = [norm(x) for x in syn]
                self._search.append((item, norm(name), syns, (norm(name) + " " + " ".join(syns)).split()))
                for phrase in [name, *syn]:
                    for k in _variants(phrase):
                        if k not in self.phrase or n > self.phrase[k]["diseases"]:
                            self.phrase[k] = item
                            self.max_words = max(self.max_words, min(len(k.split()), 8))
            # everyday terms that have no HPO match still get detected in text
            for key, label, term, cat, hpo in COMMON_SYMPTOMS:
                target = self.by_id.get(hpo) or {"id": f"common:{key}", "name": term, "synonyms": [],
                                                 "category": cat, "diseases": 0, "common": True}
                for phrase in (label, term):
                    for k in _variants(phrase):
                        self.phrase.setdefault(k, target)
            for said, clinical in COLLOQUIAL.items():
                target = self.phrase.get(stem_phrase(clinical))
                if target:
                    self.phrase.setdefault(stem_phrase(said), target)
            self._loaded = True
        return self

    # ------------------------------------------------------------------
    def search(self, q: str, limit: int = 20) -> list[dict]:
        self.load()
        qn = norm(q)
        if not qn:
            return []
        words = qn.split()
        out: list[tuple[tuple, dict]] = []
        seen = set()

        def add(score, item, matched):
            if item["id"] in seen:
                return
            seen.add(item["id"])
            out.append((score, {**item, "matched": matched}))

        # everyday symptoms first (plain-language labels)
        for key, label, term, cat, hpo in COMMON_SYMPTOMS:
            for text in (label, term):
                tn = norm(text)
                if tn.startswith(qn) or all(any(t.startswith(w) for t in tn.split()) for w in words):
                    base = self.by_id.get(hpo)
                    item = {"id": hpo or f"common:{key}", "name": term, "label": label,
                            "synonyms": [], "category": cat, "diseases": base["diseases"] if base else 0,
                            "common": True}
                    add((0, 0 if tn.startswith(qn) else 1, len(tn)), item, label)
                    break

        for item, name, syns, toks in self._search:
            if name.startswith(qn):
                add((1, 0, -item["diseases"], len(name)), item, item["name"])
                continue
            hit = next((item["synonyms"][i] for i, sn in enumerate(syns) if sn.startswith(qn)), None)
            if hit:
                add((2, 0, -item["diseases"], len(name)), item, hit)
            elif all(any(t.startswith(w) for t in toks) for w in words):
                add((3, 0, -item["diseases"], len(name)), item, item["name"])
            elif len(qn) >= 5 and qn in " ".join(toks):
                add((4, 0, -item["diseases"], len(name)), item, item["name"])
            if len(out) > 400:
                break
        out.sort(key=lambda x: x[0])
        return [x[1] for x in out[:limit]]

    def find_in_text(self, text: str, negation: bool = True) -> list[dict]:
        """Phenotypes mentioned in free text (longest match first, no overlaps).
        With negation=True, findings within a few words after "no", "not", "without",
        "denies" (in the same sentence) are skipped: "no fever or cough" finds nothing."""
        self.load()
        toks, neg_until = [], []
        for sent in re.split(r"[.;:\n]+", text.replace("'", "")):
            st = stem_phrase(sent).split()
            last_neg = -99
            for j, t in enumerate(st):
                if t in NEGATORS:
                    last_neg = j
                elif t in CONTRAST:
                    last_neg = -99
                toks.append(t)
                neg_until.append(negation and 0 < j - last_neg <= NEG_WINDOW)
        found, used = [], set()
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(self.max_words, len(toks) - i), 0, -1):
                phrase = " ".join(toks[i:i + n])
                item = self.phrase.get(phrase)
                if item and not (n == 1 and len(phrase) < 4):
                    hit = (n, item, phrase)
                    break
            if hit:
                n, item, phrase = hit
                if neg_until[i]:
                    i += n
                    continue
                if item["id"] not in used:
                    used.add(item["id"])
                    found.append({**item, "matched": phrase})
                i += n
            else:
                i += 1
        return found


NEGATORS = {"no", "not", "without", "denie", "denies", "denied", "negative", "nor", "never"}
NEG_WINDOW = 4
CONTRAST = {"but", "however", "although", "yet", "with", "has", "have"}

# Everyday phrasings that the phenotype index does not cover, mapped straight to the
# everyday-symptom keys in clinical.COMMON_SYMPTOMS. Lower-case regexes.
EVERYDAY_PATTERNS = {
    "constipation": r"constipat|hard,? (?:infrequent |dry )?stools?|infrequent stools?|straining|stools? (?:only )?(?:every|once|twice) \w+ days?|not opened (?:his|her|my) bowels",
    "blood_stool": r"blood (?:on|in) (?:the )?(?:toilet )?(?:paper|stools?|pan)|rectal bleeding|bleeding from the (?:back passage|bottom|anus)",
    "diarrhea": r"loose (?:motions?|stools?)|watery stools?|diarrh",
    "knee_pain": r"(?:painful|aching|sore|stiff) knees?|knees? (?:pain|ache|aching|hurt)",
    "joint_pain": r"big toe|toe joint|(?:painful|aching|sore) (?:joints?|knuckles?|wrists?|fingers?)|joints? (?:pain|ache)",
    "joint_swelling": r"swollen (?:big )?(?:toe|joints?|knuckles?|wrists?|fingers?|knees?|ankles? joint)|(?:red|hot),? (?:hot,? )?swollen (?:big )?toe",
    "morning_stiffness": r"(?:stiff|stiffness)[^.]{0,30}morning|morning stiffness",
    "acne": r"pimples?|blackheads?|whiteheads?|\bzits?\b|pustules? on (?:the |his |her )?(?:face|forehead|chin|back)",
    "chest_pain": r"chest (?:pain|tightness|pressure|discomfort|heaviness)|tight(?:ness)? (?:in the )?chest",
    "breathless": r"breathless|short(?:ness)? of breath|out of breath|gasping",
    "high_bp": r"(?:blood pressure|\bbp\b)[^.]{0,25}?\b1[4-9]\d\s*/\s*\d{2,3}|high blood pressure|raised blood pressure|hypertensi",
    "poor_sleep": r"can'?t sleep|cannot sleep|lies? awake|trouble (?:sleeping|falling asleep)|difficulty (?:sleeping|falling asleep)|wak(?:es|ing) (?:up )?(?:early|at [2-5])|insomnia|poor sleep",
    "low_mood": r"low mood|feeling (?:sad|low|down|empty|hopeless|worthless)|no interest|lost interest|loss of interest",
    "anxiety": r"worr(?:y|ies|ying)|anxious|nervous|on edge|panic",
    "toothache": r"tooth ?ache|tooth pain|painful tooth|teeth (?:pain|hurt)",
    "mouth_ulcers": r"(?:mouth|oral) ulcers?|ulcers? (?:inside|in) (?:the |his |her |my )?mouth|canker sores?",
    "skin_infection": r"red,? hot,? (?:and )?swollen|spreading redness|hot,? red,? (?:and )?swollen (?:leg|arm|skin)",
    "flank_pain": r"\bloin\b|flank pain|pain in the (?:right|left) (?:loin|flank|side)",
    "frequent_urination": r"going (?:to the toilet )?(?:very )?often|peeing (?:a lot|often)|urinat\w* (?:very )?often|urgency|urgently",
    "painful_urination": r"burning (?:when|on) (?:passing urine|peeing|urinating)|stinging (?:when|on) (?:peeing|passing urine)|painful urination|dysuria",
    "vaginal_discharge": r"vaginal discharge|thick white discharge|(?:white|grey|fishy) discharge",
    "period_pain": r"period pains?|painful periods|cramps?[^.]{0,30}periods?",
    "irregular_periods": r"irregular periods|periods? every (?:two|three|2|3) (?:to \w+ )?months|missed periods",
    "back_pain": r"back ?ache|(?:low(?:er)?|upper) back pain|back pain|pain in (?:the|his|her|my) (?:lower )?back",
    "neck_pain": r"neck (?:pain|stiffness)|stiff(?:,)? (?:painful )?neck|painful neck|sore neck",
    "shoulder_pain": r"shoulder (?:pain|stiffness)|(?:lifting|raising) the arm|reach(?:ing)? (?:up|behind)|comb(?:ing)? (?:her|his) hair",
    "ear_pain": r"ear ?ache|ear pain|tugging at (?:his|her|the) ear|pain(?:ful)? (?:right |left )?ear",
    "itching": r"\bitch",
    "rash": r"\brash\b|spots? (?:on|all over)|welts?",
    "fatigue": r"tired|exhausted|no energy|fatigue|worn out",
    "fever": r"fever|feverish|high temperature|\bpyrexi|shiver",
    "runny_nose": r"runny nose|running nose|nasal discharge|rhinorrh",
    "blocked_nose": r"blocked nose|stuffy nose|nasal congestion|congested nose",
    "sore_throat": r"sore throat|throat (?:pain|is sore)|painful swallowing|pain on swallowing|scratchy throat",
    "headache": r"headaches?|head ?ache",
    "wheeze": r"wheez",
    "red_eye": r"red(?:,)? (?:itchy )?(?:watery )?eyes?|pink (?:sticky )?eye|conjunctiv",
    "numbness": r"pins and needles|tingling|numb",
}
_EVERYDAY_RX = {k: re.compile(v) for k, v in EVERYDAY_PATTERNS.items()}
_NEG_RX = re.compile(r"\b(?:no|not|without|denies|denied|nor|never)\b")


def everyday_keys_by_pattern(text: str) -> set[str]:
    """Everyday-symptom keys found by phrase patterns, skipping negated mentions."""
    t = " " + text.lower() + " "
    keys = set()
    for key, rx in _EVERYDAY_RX.items():
        if any(not negated_at(t, m.start()) for m in rx.finditer(t)):
            keys.add(key)
    return keys


def negated_at(t: str, pos: int) -> bool:
    """True if a negator ("no", "not", "without"...) appears shortly before pos in the same
    sentence. Lists stay negated: "no fever, cough or back pain"."""
    before = re.split(r"[.;:\n]", t[max(0, pos - 60):pos])[-1]
    words = before.replace(",", " , ").split()
    tail = words[-8:]
    in_list = any(w in (",", "or", "and", "nor") for w in tail)
    window = tail if in_list else words[-NEG_WINDOW:]
    negs = [i for i, w in enumerate(window) if _NEG_RX.fullmatch(w)]
    if not negs:
        return False
    # "no fever but seizures": a contrast word after the negator ends the negation
    return not any(w in ("but", "however", "although", "yet", "with", "has", "have") for w in window[negs[-1] + 1:])


INDEX = SymptomIndex()


def common_symptoms_grouped() -> list[dict]:
    INDEX.load()
    groups: dict[str, list[dict]] = {}
    for key, label, term, cat, hpo in COMMON_SYMPTOMS:
        groups.setdefault(cat, []).append({"key": key, "label": label, "term": term, "id": hpo or f"common:{key}"})
    return [{"category": c, "items": v} for c, v in groups.items()]


def common_keys_in(text: str) -> set[str]:
    """Which everyday symptoms (clinical.COMMON_SYMPTOMS keys) a text mentions."""
    # keep sentence breaks so negation never crosses a full stop
    tn = " " + " . ".join(norm(part) for part in re.split(r"[.;:\n]+", text)) + " "
    keys = set()
    INDEX.load()
    found_ids = {f["id"] for f in INDEX.find_in_text(text)}
    for key, label, term, cat, hpo in COMMON_SYMPTOMS:
        hit = bool(hpo and hpo in found_ids)
        for phrase in (term, label):
            pn = f" {norm(phrase)} "
            i = tn.find(pn)
            while i >= 0 and not hit:
                hit = not negated_at(tn, i + 1)
                i = tn.find(pn, i + 1)
        if hit:
            keys.add(key)
    keys |= {k for k in everyday_keys_by_pattern(text) if k in COMMON_BY_KEY}
    # synonyms that share an HPO id (e.g. dizziness and vertigo) both count
    for key in list(keys):
        hpo = COMMON_BY_KEY[key][4]
        if hpo:
            keys |= {k for k, *_rest, h in COMMON_SYMPTOMS if h == hpo}
    return keys
