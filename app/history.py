"""History-taking agent: patients describe their problem before the visit.

Two modes behind one interface:
  ai      an LLM interviewer (HISTORY_SYSTEM) asks one question at a time and covers the
          standard history; a second call (SUMMARY_SYSTEM) writes the clinical summary.
  guided  a scripted interview (STEPS) with quick replies and the symptom picker. Used when
          no LLM is available, and as the automatic fallback if the LLM fails mid-way.

Every patient message is checked for red flags (clinical.RED_FLAGS); a match adds urgent-care
advice to the conversation and is highlighted for staff. The agent never diagnoses or gives
treatment advice.

The finished intake produces:
  summary   structured sections for the doctor
  case_text a clinical narrative that goes straight into the AI diagnosis page
  symptoms  phenotype terms found in the answers (OptimusKG/HPO ids) for the symptom picker
"""

from __future__ import annotations

import json
from datetime import datetime

from .clinical import COMMON_BY_KEY, EMERGENCY_ADVICE, red_flags_in
from .symptoms import INDEX

NONE_WORDS = {"none", "no", "nothing", "nil", "na", "n/a", "not really", "no thanks", "skip", "nope"}

# key, question, kind (text | choice | symptoms), options, summary label
STEPS: list[dict] = [
    {"key": "complaint", "kind": "symptoms",
     "q": "Hello {first}, I'll ask a few questions so your doctor is prepared for your visit. "
          "What is the main problem you would like help with? You can type it in your own words or pick "
          "from the list."},
    {"key": "onset", "kind": "choice",
     "q": "When did this start?",
     "options": ["Today", "1 to 3 days ago", "4 to 7 days ago", "1 to 4 weeks ago", "1 to 6 months ago",
                 "More than 6 months ago", "Since birth or early childhood"]},
    {"key": "course", "kind": "choice",
     "q": "Since it started, has it been getting better, getting worse, staying the same, or coming and going?",
     "options": ["Getting better", "Getting worse", "Staying the same", "Comes and goes"]},
    {"key": "severity", "kind": "choice",
     "q": "How much is it affecting you, from 1 (mild) to 10 (the worst you can imagine)?",
     "options": [str(i) for i in range(1, 11)]},
    {"key": "details", "kind": "text",
     "q": "Tell me a little more: where exactly is it, what does it feel like, and does anything make it "
          "better or worse? (You can say 'skip'.)"},
    {"key": "associated", "kind": "symptoms",
     "q": "Have you noticed any other symptoms? Tap all that apply, search for others, or say 'none'."},
    {"key": "pmh", "kind": "text",
     "q": "Do you have any long-term health conditions, or have you had operations or hospital stays before?"},
    {"key": "medications", "kind": "text",
     "q": "Which medicines, vitamins or herbal remedies do you take regularly? Include the dose if you know it."},
    {"key": "pregnancy", "kind": "choice", "only": "maybe_pregnant",
     "q": "Is there any chance you could be pregnant?", "options": ["No", "Yes", "Not sure"]},
    {"key": "allergies", "kind": "text",
     "q": "Are you allergic to any medicines, foods or anything else?{known_allergies}"},
    {"key": "family", "kind": "text",
     "q": "Does anyone in your family have a similar problem, or is there an illness that runs in the family? "
          "Are your parents related by blood (for example, cousins)?"},
    {"key": "social", "kind": "text",
     "q": "Do you smoke or drink alcohol? Is there anything about your work, home or recent travel that might matter?"},
    {"key": "extra", "kind": "text",
     "q": "Is there anything else you would like the doctor to know?"},
]
STEP_BY_KEY = {s["key"]: s for s in STEPS}

DONE_MESSAGE = ("Thank you. I have passed this to your doctor, who will read it before your visit. "
                "You do not need to repeat everything, but the doctor may ask follow-up questions.")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _is_none(text: str) -> bool:
    return text.strip().lower().strip(".!") in NONE_WORDS


def _applies(step: dict, patient) -> bool:
    if step.get("only") == "maybe_pregnant":
        return (patient.sex or "").lower().startswith("f") and patient.age is not None and 12 <= patient.age <= 55
    return True


def _question(step: dict, patient) -> str:
    known = f" Our records list: {patient.allergies}." if patient.allergies else ""
    return step["q"].format(first=patient.first_name, known_allergies=known)


def load(intake) -> tuple[list[dict], dict]:
    return json.loads(intake.messages_json or "[]"), json.loads(intake.answers_json or "{}")


def save(intake, messages: list[dict], answers: dict) -> None:
    intake.messages_json = json.dumps(messages, ensure_ascii=False)
    intake.answers_json = json.dumps(answers, ensure_ascii=False)


def _msg(role: str, text: str, **extra) -> dict:
    return {"role": role, "text": text, "at": datetime.now().strftime("%H:%M"), **extra}


def current_step(intake, patient) -> dict | None:
    if intake.mode != "guided" or intake.status != "in_progress":
        return None
    i = intake.step
    while i < len(STEPS) and not _applies(STEPS[i], patient):
        i += 1
    return STEPS[i] if i < len(STEPS) else None


# ----------------------------------------------------------------------------
# start / reply
# ----------------------------------------------------------------------------
def start(intake, patient, llm=None) -> None:
    messages: list[dict] = []
    if intake.mode == "ai" and llm is not None:
        try:
            reply = _ai_turn(llm, patient, messages)
            messages.append(_msg("assistant", reply["reply"], quick=reply.get("quick_replies") or []))
            save(intake, messages, {})
            return
        except Exception:
            intake.mode = "guided"
    intake.mode, intake.step = "guided", 0
    messages.append(_msg("assistant", _question(STEPS[0], patient), step="complaint"))
    save(intake, messages, {})


def reply(intake, patient, text: str, picked: list[dict] | None = None, llm=None) -> None:
    """Record the patient's answer and append the agent's next message(s)."""
    messages, answers = load(intake)
    picked = picked or []
    shown = text.strip()
    if picked:
        names = ", ".join(p.get("label") or p["name"] for p in picked)
        shown = f"{shown}. {names}" if shown and not _is_none(shown) else names
    if not shown:
        return
    messages.append(_msg("user", shown))

    flags = red_flags_in(shown)
    known = set(filter(None, (intake.red_flags or "").split("; ")))
    new = [f for f in flags if f not in known]
    if new:
        intake.red_flags = "; ".join(sorted(known | set(new)))
        messages.append(_msg("assistant", f"You mentioned {', '.join(new)}. {EMERGENCY_ADVICE} "
                             "If it is not urgent, we can carry on.", urgent=True))

    if intake.mode == "ai" and llm is not None:
        try:
            r = _ai_turn(llm, patient, messages)
            if r.get("done"):
                _finish(intake, patient, messages, answers, llm)
            else:
                messages.append(_msg("assistant", r["reply"], quick=r.get("quick_replies") or []))
            save(intake, messages, answers)
            return
        except Exception:
            # fall back to the guided interview without losing what was said
            intake.mode = "guided"
            first = next((m["text"] for m in messages if m["role"] == "user"), "")
            answers.setdefault("complaint", {"text": first, "picked": picked})
            intake.step = 1
            messages.append(_msg("assistant", "Let me continue with a few set questions."))
            step = current_step(intake, patient)
            if step:
                messages.append(_msg("assistant", _question(step, patient), step=step["key"]))
            save(intake, messages, answers)
            return

    step = current_step(intake, patient)
    if step is None:
        _finish(intake, patient, messages, answers, None)
        save(intake, messages, answers)
        return
    answers[step["key"]] = {"text": text.strip(), "picked": picked}
    intake.step = STEPS.index(step) + 1
    nxt = current_step(intake, patient)
    if nxt is None:
        _finish(intake, patient, messages, answers, None)
    else:
        messages.append(_msg("assistant", _question(nxt, patient), step=nxt["key"]))
    save(intake, messages, answers)


def _finish(intake, patient, messages, answers, llm) -> None:
    summary = None
    if llm is not None and intake.mode == "ai":
        try:
            summary = _ai_summary(llm, patient, messages)
        except Exception:
            summary = None
    if summary is None:
        summary = (guided_summary(patient, answers) if answers else transcript_summary(patient, messages))
    patient_text = " ".join(m["text"] for m in messages if m["role"] == "user")
    terms = INDEX.find_in_text(patient_text + " " + summary.get("case_text", ""))
    for a in answers.values():
        for p in a.get("picked") or []:
            if p.get("id") and all(t["id"] != p["id"] for t in terms):
                terms.append({"id": p["id"], "name": p.get("name") or p.get("label")})
    summary["symptoms"] = [{"id": t["id"], "name": t["name"]} for t in terms][:30]
    summary["red_flags"] = [f for f in (intake.red_flags or "").split("; ") if f]
    photos = [m for m in messages if m["role"] == "photo"]
    if photos:
        summary.setdefault("sections", {})["Photos"] = (
            f"{len(photos)} photo{'s' if len(photos) != 1 else ''} attached by the patient: "
            + "; ".join(m["text"].split(": ", 1)[1] if ": " in m["text"] else m["text"] for m in photos))
        summary["image_ids"] = [m["image_id"] for m in photos if m.get("image_id")]
    intake.summary_json = json.dumps(summary, ensure_ascii=False)
    intake.case_text = summary.get("case_text", "")
    intake.status, intake.completed_at = "completed", datetime.now()
    messages.append(_msg("assistant", DONE_MESSAGE, done=True))


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------
DURATION = {"Today": "since today", "1 to 3 days ago": "for 1-3 days", "4 to 7 days ago": "for 4-7 days",
            "1 to 4 weeks ago": "for 1-4 weeks", "1 to 6 months ago": "for 1-6 months",
            "More than 6 months ago": "for more than 6 months",
            "Since birth or early childhood": "since birth or early childhood"}


def _ans(answers: dict, key: str) -> str:
    a = answers.get(key) or {}
    parts = [a.get("text", "").strip()] if a.get("text") and not _is_none(a.get("text", "")) else []
    parts += [p.get("name") or p.get("label") for p in a.get("picked") or []]
    return "; ".join(x for x in parts if x)


def _who(patient) -> str:
    sex = (patient.sex or "").lower()
    age = patient.age
    if age is None:
        noun = {"f": "woman", "m": "man"}.get(sex[:1], "patient")
        return f"A {noun}"
    if age < 18:
        noun = {"f": "girl", "m": "boy"}.get(sex[:1], "child")
    else:
        noun = {"f": "woman", "m": "man"}.get(sex[:1], "patient")
    return f"A {age}-year-old {noun}"


def guided_summary(patient, answers: dict) -> dict:
    complaint = _ans(answers, "complaint") or "an unspecified problem"
    onset = (answers.get("onset") or {}).get("text", "")
    course = (answers.get("course") or {}).get("text", "")
    sev = (answers.get("severity") or {}).get("text", "")
    sections = {
        "Main problem": complaint,
        "Started": onset,
        "Course": course,
        "Severity": f"{sev}/10" if sev.isdigit() else sev,
        "Details": _ans(answers, "details"),
        "Other symptoms": _ans(answers, "associated") or "None reported",
        "Medical history": _ans(answers, "pmh") or "None reported",
        "Medicines": _ans(answers, "medications") or "None reported",
        "Pregnancy": _ans(answers, "pregnancy"),
        "Allergies": _ans(answers, "allergies") or (patient.allergies or "None reported"),
        "Family history": _ans(answers, "family") or "None reported",
        "Social": _ans(answers, "social"),
        "Other information": _ans(answers, "extra"),
    }
    sections = {k: v for k, v in sections.items() if v}

    c = complaint.rstrip(".")
    if c[:1].isupper() and c[1:2].islower():   # "Headache" -> "headache", keep "HIV", "ECG"
        c = c[0].lower() + c[1:]
    s = [f"{_who(patient)} presenting with {c}"]
    if onset:
        s[0] += f" {DURATION.get(onset, onset.lower())}"
    s[0] += "."
    if course:
        s.append(f"It is {course.lower()}" + (f", severity {sev}/10." if sev else "."))
    if sections.get("Details"):
        s.append(sections["Details"].rstrip(".") + ".")
    if _ans(answers, "associated"):
        s.append(f"Associated symptoms: {_ans(answers, 'associated')}.")
    if _ans(answers, "pmh"):
        s.append(f"Past medical history: {_ans(answers, 'pmh')}.")
    if _ans(answers, "medications"):
        s.append(f"Medications: {_ans(answers, 'medications')}.")
    if sections.get("Allergies") and sections["Allergies"] != "None reported":
        s.append(f"Allergies: {sections['Allergies']}.")
    if _ans(answers, "pregnancy") and _ans(answers, "pregnancy").lower() != "no":
        s.append(f"Possible pregnancy: {_ans(answers, 'pregnancy').lower()}.")
    if _ans(answers, "family"):
        s.append(f"Family history: {_ans(answers, 'family')}.")
    if _ans(answers, "social"):
        s.append(f"Social history: {_ans(answers, 'social')}.")
    if _ans(answers, "extra"):
        s.append(f"Also reports: {_ans(answers, 'extra')}.")
    return {"sections": sections, "case_text": " ".join(s), "source": "guided"}


def transcript_summary(patient, messages: list[dict]) -> dict:
    said = [m["text"] for m in messages if m["role"] == "user"]
    text = f"{_who(patient)}. Patient's own account: " + " ".join(x.rstrip(".") + "." for x in said)
    return {"sections": {"Patient's account": " / ".join(said)}, "case_text": text, "source": "transcript"}


# ----------------------------------------------------------------------------
# LLM mode
# ----------------------------------------------------------------------------
HISTORY_SYSTEM = """You are a friendly clinic intake assistant taking a medical history from a
patient (or a parent answering for a child) before their appointment. Write in plain, warm
language at a reading age of about 12.

Rules:
- Ask ONE short question at a time. Do not repeat questions already answered.
- Cover, in a natural order: the main problem; onset and duration; course; location, character,
  severity (1-10), timing, triggers and relieving factors; associated symptoms; relevant review
  of systems; past medical and surgical history; current medicines; allergies; pregnancy if
  relevant; family history (including similar illness in relatives and whether parents are
  related by blood); social history (smoking, alcohol, occupation, travel); for children, birth
  and developmental history; anything else the patient wants the doctor to know.
- Never give a diagnosis, never say what the condition might be, never suggest treatment or
  medicines. If asked, say the doctor will discuss this at the visit.
- If the patient describes an emergency (for example chest pain, trouble breathing, stroke signs,
  severe bleeding, suicidal thoughts), tell them to call 112 or go to the nearest emergency
  department now, then continue only if they say it is not urgent.
- Finish when the history is reasonably complete (usually 10-16 questions) or the patient wants
  to stop.

Return ONLY JSON: {"reply": "your next message", "quick_replies": ["up to 6 short suggested
answers, or empty"], "done": false}. When finished return {"reply": "", "quick_replies": [],
"done": true}."""

SUMMARY_SYSTEM = """You are a clinician writing up a patient-reported history for the treating
doctor. Use only what the patient said; do not add findings, diagnoses or advice. Keep the
patient's uncertainty ("not sure", "maybe") where relevant.

Return ONLY JSON:
{"sections": {"Main problem": "", "History of presenting complaint": "", "Other symptoms": "",
  "Medical history": "", "Medicines": "", "Allergies": "", "Family history": "", "Social": "",
  "Other information": ""},
 "case_text": "one clinical paragraph, third person, starting with age and sex, e.g. 'A 34-year-old
  woman presenting with ...', including pertinent negatives the patient reported"}"""


def _context(patient) -> str:
    bits = [f"Patient: {patient.first_name}", f"age {patient.age}" if patient.age is not None else "age unknown",
            f"sex {patient.sex or 'unknown'}"]
    if patient.allergies:
        bits.append(f"allergies on record: {patient.allergies}")
    return ", ".join(bits)


def _ai_turn(llm, patient, messages: list[dict]) -> dict:
    convo = [{"role": "system", "content": HISTORY_SYSTEM + "\n\n" + _context(patient)}]
    for m in messages:
        if m["role"] == "photo":   # the model cannot see images; it only knows one was attached
            convo.append({"role": "user", "content": f"(The patient attached a photo for the doctor. {m['text']})"})
            continue
        convo.append({"role": "assistant" if m["role"] == "assistant" else "user", "content": m["text"]})
    if len(messages) == 0:
        convo.append({"role": "user", "content": "(The patient has opened the intake. Greet them by first "
                                                 "name and ask the first question.)"})
    if sum(1 for m in messages if m["role"] == "user") >= 24:
        return {"reply": "", "done": True}
    obj, meta = llm.json_chat(convo, max_tokens=400, required=["reply"], tag="history")
    if not obj:
        raise RuntimeError(meta.get("error") or "no reply")
    done = bool(obj.get("done"))
    text = str(obj.get("reply", "")).strip()
    if not done and not text:
        raise RuntimeError("empty reply")
    quick = [str(x)[:60] for x in (obj.get("quick_replies") or []) if str(x).strip()][:6]
    return {"reply": text, "quick_replies": quick, "done": done}


def _ai_summary(llm, patient, messages: list[dict]) -> dict:
    transcript = "\n".join(f"{'Assistant' if m['role'] == 'assistant' else 'Patient'}: {m['text']}"
                           for m in messages)
    obj, meta = llm.json_chat([{"role": "system", "content": SUMMARY_SYSTEM},
                               {"role": "user", "content": _context(patient) + "\n\nTRANSCRIPT:\n" + transcript
                                + "\n\nReturn the JSON object."}],
                              max_tokens=1200, required=["case_text"], tag="history-summary")
    if not obj or not str(obj.get("case_text", "")).strip():
        raise RuntimeError(meta.get("error") or "no summary")
    sections = {k: str(v).strip() for k, v in (obj.get("sections") or {}).items() if str(v).strip()}
    return {"sections": sections, "case_text": str(obj["case_text"]).strip(), "source": "ai"}


def common_symptom_choices() -> list[dict]:
    return [{"key": k, "label": v[1], "term": v[2], "id": v[4] or f"common:{k}"} for k, v in COMMON_BY_KEY.items()]
