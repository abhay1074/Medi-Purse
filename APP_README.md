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

## What's in it

| Role | Can do |
|---|---|
| Registrar / Admin | Register hospitals, clinics (in a hospital or independent), staff accounts; sees everything |
| Doctor | Patients in their clinic, appointments, diagnoses, medicines, AI diagnosis, medicine reports |
| Receptionist | Register patients, book and manage appointments, record feedback, medicine reports |

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
- **Feedback**: 1-5 rating plus comments per patient or visit; clinic-wide list with average.

## AI diagnosis

1. **Find likely conditions** (free, local, about 1 s): the fusion v4 retriever ranks
   1,464 rare diseases and shows matched case terms, Orphanet profile sentences,
   inheritance/onset/prevalence, and known phenotypes and genes for each.
2. **Explain with AI** (optional, DeepSeek, 4 calls, about 30-90 s): extracts findings, audits
   each suggestion against the evidence with ranks hidden, writes cited claims, and
   adjudicates every citation. Dropped claims are hidden and softened ones reworded.
3. **Add to patient's diagnoses**: the doctor confirms and records the condition, linked
   to the analysis.

The retriever's order always stays the primary ranking. The bundle's evaluation found
that letting the LLM reorder hurts correct top-1 cases, so the LLM's plausibility score is shown
alongside instead.

Code: `app/ai.py` wraps `handoff/code/` (the explain pipeline is adapted from
`live_engine.py`; it skips the LLM-only baseline call). Provider settings are in `.env`.
Any OpenAI-compatible API works by changing `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.

**Research prototype, not a diagnostic device.** Use made-up or published cases only;
do not send real patient data to an external API.

## Layout

```
app/main.py        routes and access rules
app/db.py          SQLite models
app/ai.py          AI service (local retriever + DeepSeek explanation)
app/seed.py        demo data
app/templates/     pages (Jinja2)
app/static/        style.css
handoff/           Aarav's bundle: weights, catalog, pipeline code, docs
```
