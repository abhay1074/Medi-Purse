# MediPurse: hospital management with AI diagnosis support

A hospital management system built on the Glassbox rare-disease retriever in `handoff/`.
The UI follows shadcn/ui design tokens (Inter, zinc greys, blue primary, Lucide icons) in plain CSS.

## Run it (Windows)

```powershell
uv venv .venv
uv pip install -p .venv -r requirements.txt
copy .env.example .env      # then put your DeepSeek key in .env
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000. On first start the database (`hospital.db`) is created
with demo data. Delete `hospital.db` to start fresh.

Demo accounts (password is `DEMO_PASSWORD` in `app/seed.py`):

| Email | Role |
|---|---|
| admin@medipurse.test | Registrar / Admin |
| dr.mehta@medipurse.test | Doctor, Genetics Clinic |
| dr.rao@medipurse.test | Doctor, Neurology Clinic |
| reception@medipurse.test | Receptionist, Genetics Clinic |
| pharmacy@medipurse.test | Pharmacist, Genetics Clinic |

Patient portal demo: sign in at http://localhost:8000/portal with sunita.verma@example.test
(same demo password).

## What's in it

Visitors who are not signed in see a public landing page at `/` (also `/welcome`) with links to
book, the patient portal and staff sign-in. Signed-in staff get their dashboard; patients their portal.

| Role | Can do |
|---|---|
| Registrar / Admin | Register hospitals, clinics (in a hospital or independent), staff accounts; sees everything |
| Doctor | Patients in their clinic, appointments, diagnoses, medicines, AI diagnosis, medicine reports |
| Receptionist | Register patients, book and manage appointments, record feedback, give prescribed medicines, medicine reports (manages stock only if the clinic has no pharmacist) |
| Pharmacist | Manages the clinic's stock and prices, gives prescribed medicines, sells over the counter with receipts, medicine reports. No access to patient records, appointments or AI |

- **Patients**: registration, search (name, phone or ID like `P-000001`), record page with
  tabs for overview, active diagnoses and diagnosis history, upcoming appointments and
  appointment history, AI analyses, and feedback.
- **Appointments**: booking with a 15-minute double-booking check per doctor, status
  (scheduled, completed, cancelled, no-show), consultation notes, filters by date/status/doctor.
- **Medicines**: doctors prescribe one or more medicines at once (dosage, frequency,
  duration, instructions), linked to a diagnosis and visit; stop or restart them later.
  Patient record has a Medicines tab with current medicines and history.
- **Medicine reports**: pick Today, Last 2 days, Last 7 days, Last 30 days or custom dates
  (and optionally a doctor), preview, then **Download Excel**. The .xlsx has date, time,
  patient, disease, medicine, dosage, frequency, duration, doctor and clinic, plus the
  period and a generated-at timestamp; the file name includes both.
- **Patient portal** (`/portal`): patients create an account (or accept an emailed invite
  from staff), book with a doctor in their clinic by choosing a day and a free 30-minute slot
  (Mon-Sat 9:00-17:00, lunch 13:00-14:00 closed, up to 30 days ahead, at most 3 upcoming),
  cancel up to 2 hours before, see current medicines, and turn email reminders on or off.
- **Emails**: confirmation with a calendar invite (.ics) when booked by staff or online,
  emails when an appointment is moved or cancelled, automatic reminders 24 h and 2 h before
  (`REMINDER_HOURS`), portal invites and password links. A "Send reminder now" button is on
  each appointment. Doctors get an email when a patient books online.
- **Alerts**: new online bookings show on the dashboard with a red badge in the menu until
  staff mark them seen. The **Email log** page lists every email with its status, and failed
  ones can be retried.
- **Feedback**: 1-5 rating plus comments per patient or visit; clinic-wide list with average.

## Pharmacy

Who does what:

| Role | Pharmacy work |
| --- | --- |
| Doctor | Prescribes (sees what is in stock, never changes stock) |
| Pharmacist | Manages stock (add, receive, correct, prices), gives prescribed medicines, counter sales |
| Receptionist | Gives prescribed medicines; manages stock only when the clinic has no pharmacist |
| Admin | Everything, every clinic |

1. **Prescribe** (doctor): tick common medicines or type any medicine; dose, frequency and duration
   are dropdowns and the quantity for the course is worked out (tablets per dose x times a day x
   days; syrups, inhalers and creams count as one pack). Nothing is taken from stock yet; the
   prescription goes to the **Dispensing** queue and the patient's email says what to collect at
   the clinic and what to buy elsewhere.
2. **Give** (pharmacist or receptionist): open the prescription in Dispensing, check or change
   the stock item and quantity per line, and save. Only then is stock deducted. Anything not given
   is marked "buy from another pharmacy"; expired stock is never given. The patient is emailed
   what they received.
3. **Counter sales** (pharmacist): walk-in customers or prescriptions from outside the clinic.
   Choose medicines and quantities, prices come from stock (editable), payment cash/UPI/card,
   stock is deducted, and a printable receipt is created. Daily totals by payment method.

**Pharmacy stock** shows each clinic's medicines with stock, price, reorder level, batch and
expiry, with filters for low, out of stock and expiring within 60 days. Every change (received,
given on a prescription, sold, corrected) is kept in the medicine's stock history.

Light and dark mode follow the device setting; the sun/moon button switches and remembers it.

## Prescribing: suggestions and reactions

- **Suggestions for the chosen diagnosis** (`app/drug_suggest.py`):
  - **Everyday diagnoses:** usual first-line options with typical doses.
  - **Rare diagnoses:** the medicines used in published ZebraMap case reports for that disease,
    with how many cases used each, plus approved indications from OptimusKG.
    The data is in `app/data/disease_drugs.json`, built by `scripts/build_drug_suggestions.py`.
- **Contraindications:** if a medicine is listed as contraindicated in one of the patient's
  diagnoses (OptimusKG), the line is flagged.
- **Reactions:** "Report reaction" on any medicine records what happened (irritation, rash,
  allergy…), the severity and notes. It can stop the medicine and mark it **to avoid**.
  Medicines given elsewhere can be recorded too.
- **The avoid list** shows as a red banner on the patient, prescription and dispensing pages.
- **Prescribing an avoided medicine** needs explicit confirmation. Brand names and abbreviations
  are matched to the generic: pcm → paracetamol, Augmentin → amoxicillin.
- **Lifting a flag:** doctors can remove an avoid flag; the reaction stays in the record.

## Email setup

What is emailed (patients can turn emails off in "My details"; the Email log shows all of them):

| To | When |
| --- | --- |
| Patient | Appointment booked (with calendar invite), moved, cancelled; reminders 24 h and 2 h before |
| Patient | New prescription (each medicine with dose, frequency, duration, instructions); medicine stopped |
| Patient | Feedback request after a diagnosis is recorded or a visit is marked completed (one per visit; a one-time link, valid 14 days, no sign-in; the diagnosis itself is not put in the email) |
| Patient | Portal invite, password link |
| Doctor | New online booking; urgent symptoms (red flags) in a patient check-in; feedback rated 1 or 2 |

Emails are queued and sent by a background worker in the app. Put SMTP settings in `.env`
(see `.env.example`). For Gmail, turn on 2-step verification and create an App password,
then use `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`, `SMTP_USER` = your address and
`SMTP_PASSWORD` = the app password. Without `SMTP_HOST`, emails are saved as `.eml` files in
`outbox/` (open them in Outlook or any mail program) so everything can be tried locally.
Set `APP_BASE_URL` to the address patients use to reach the app, so links in emails work.

## AI diagnosis

0. **Patient history (history-taking agent)**: before a visit, the patient opens
   "Tell us about your symptoms" in the portal and answers one question at a time: main
   problem, onset, course, severity, details, other symptoms (with the symptom picker),
   medical history, medicines, pregnancy where relevant, allergies, family history
   (including consanguinity) and social history. Every message is checked for red flags
   (chest pain, breathing trouble, stroke signs, suicidal thoughts...) and the patient is
   told to call 112 if urgent. The result is a structured summary, a clinical paragraph and
   the HPO terms found. Doctors see new histories on the dashboard, on the patient's
   "Patient history" tab and on the appointment, and "Use in AI diagnosis" fills the case.
   With an LLM it is a free-form AI interviewer (`app/history.py`, `HISTORY_SYSTEM`); without
   one, or if the LLM fails mid-way, it continues as a guided interview.
1. **Symptoms and signs**: on the AI diagnosis page, search the full OptimusKG phenotype list
   (18,555 terms: every HPO "phenotypic abnormality" term plus MedDRA symptoms, with synonyms)
   or tap everyday symptoms (`app/clinical.py`). Chosen terms are added to the case text.
   **Images** (up to 4 per case, the pipeline's limit): pick from the patient's record or upload
   on the spot, and write what each shows. The description is added to the case text
   ("Imaging: Fundus photograph (right eye): angioid streaks ...") so the ranker and realist read
   it, and the explanation step receives the images as metadata and cites each one as an imaging
   finding, as in `handoff/code/live_engine.py`. Descriptions a patient wrote are marked as theirs.
2. **Realist agent** (`app/realist.py`): reads every case summary and decides whether an
   upfront rare-disease workup is reasonable. For simple or everyday presentations it lists
   the 5 most likely common conditions, shown first (C1-C5) with the rare list below for
   reference; otherwise common causes to rule out are added after the rare list. It runs
   instantly with rules (phenotype detection, rare-disease cues, symptom overlap) and is
   refined by one LLM call when an LLM with credit is available.
3. **Find likely conditions** (free, local, about 1 s): the fusion v4 retriever ranks
   1,464 rare diseases and shows matched case terms, Orphanet profile sentences,
   inheritance/onset/prevalence, and known phenotypes and genes for each.
4. **Explain with AI** (optional, DeepSeek, 4 calls, about 30-90 s): extracts findings, audits
   each suggestion against the evidence with ranks hidden, writes cited claims, and
   adjudicates every citation. Dropped claims are hidden and softened ones reworded.
5. **Add to patient's diagnoses**: the doctor confirms and records the condition, linked
   to the analysis.

The retriever's order always stays the primary ranking. The bundle's evaluation found
that letting the LLM reorder hurts correct top-1 cases, so the LLM's plausibility score is shown
alongside instead.

LLM features check the DeepSeek balance (free endpoint, cached 10 minutes) and switch to
their rule-based or guided modes when there is no credit, so nothing breaks.

**Phenotype list**: `app/data/phenotypes.json` is built from OptimusKG by
`scripts/build_phenotypes.py`, which needs the `optimuskg` package (Python 3.12+, so it runs in
a separate `.venv-kg`; the app itself stays on 3.11). See the script for how to rebuild it.

Code: `app/ai.py` wraps `handoff/code/` (the explain pipeline is adapted from
`live_engine.py`; it skips the LLM-only baseline call). Provider settings are in `.env`.
Any OpenAI-compatible API works by changing `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.

## Clinical images

Staff upload photos, X-rays, scans, ECGs and reports on a patient's **Images** tab (or straight
from the AI diagnosis page). Patients can add photos while talking to the history agent
("Add a photo for your doctor"); they show in the conversation, on the history page, and are
pre-selected when the doctor uses that history in an AI diagnosis.

How images are stored (`app/images.py`):

- In `uploads/p<patient id>/` (or `HMS_UPLOADS`), outside the web root and git. They are served
  only by `/images/<id>`, which checks the viewer is staff of the patient's clinic (or an admin)
  or the patient. Others get 404. Responses are `private`, `nosniff`, and sandboxed by CSP.
- Every upload is decoded and re-encoded with Pillow: non-images are rejected, EXIF (GPS
  location, device, names) is removed, camera rotation is applied, and images are capped at
  3000 px. Photos stay JPEG; scans with transparency or 16-bit depth become PNG. A 360 px
  thumbnail is made for lists. Limit `HMS_MAX_UPLOAD_MB` (default 15) per file.
- Files are named by content hash, so names never contain patient details, and the same photo
  uploaded twice for one patient is stored once.
- Removing an image hides it (doctors, admins or whoever uploaded it). The file stays for the
  record; `images.purge_deleted(db)` deletes files of removed images. Back up `uploads/`
  together with `hospital.db`.
- Optional machine captions use the same BLIP radiology model as the handoff pipeline:
  set `HMS_IMAGE_CAPTIONS=1` and install `torch` and `transformers` (large). Off by default;
  the pipeline treats captions as unreliable and the written description as the finding.

**Evaluation:** `evaluation/README.md` has two parts.

- **Real-world usage:**
  - results in about 0.33 s for one doctor (p99 0.41 s);
  - about 2.9 analyses per second per server;
  - ready 9 s after start, using about 0.6 GB;
  - "Explain with AI" about 17 s and about ₹1 per case;
  - coverage of the knowledge behind the suggestions.
- **Accuracy verification:**
  - the official test split (MRR 0.496, top-10 0.685);
  - 92 synthetic cases written as case reports: combined MRR@10 0.832 with the router, against
    0.518 with the old rules;
  - 300 real held-out case reports.

Doctors and admins see a summary under AI diagnosis → "How accurate is this?" (`/ai/performance`).

**Realist router** (`app/router.py`): a small logistic model fitted by `scripts/fit_router.py`
decides whether common causes come first. It combines the rare-disease ranking's confidence with
text signals: specialised tests mentioned, how well symptoms fit a common condition, and
rare-disease cues. Without `app/data/router.json`, the original rules are used.

**Research prototype, not a diagnostic device.** Use made-up or published cases only;
do not send real patient data to an external API.

## Layout

```
app/main.py        routes and access rules
app/db.py          SQLite models
app/ai.py          AI service (local retriever + DeepSeek explanation)
app/seed.py        demo data
app/realist.py     realist agent (common vs rare)
app/history.py     history-taking agent
app/symptoms.py    symptom / phenotype search
app/clinical.py    everyday symptoms, common conditions, red flags
app/images.py      clinical image storage and AI pipeline glue
uploads/           stored images (not in git)
app/data/          OptimusKG phenotype list
scripts/           build_phenotypes.py
evaluation/        synthetic test cases, run_eval.py, results.json, README.md
app/templates/     pages (Jinja2)
app/static/        style.css
handoff/           Aarav's bundle: weights, catalog, pipeline code, docs
```
