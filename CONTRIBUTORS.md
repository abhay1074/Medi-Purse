# Contributors

MediPurse is a three-person team project. The commit history is maintained by Abhay Yadav, so it
does not reflect how the work was split — all three contributors built the system together.

| Contributor | Areas |
|---|---|
| **Abhay Yadav** | Clinic and hospital platform, patient portal, pharmacy and dispensing, appointments and reminders, reports, UI |
| **Aarav Singh** | Rare-disease retrieval (Glassbox fusion v4 SHAP-8), seed data and retriever bundle, initial explanation engine and architecture notes (`handoff/`) |
| **Animesh Pathak** | Realist router (`app/router.py`, `scripts/fit_router.py`), FastAPI backend and frontend/LLM integration (`app/main.py`, `app/ai.py`), four-call explanation refinement with citation validation, evaluation (`evaluation/`), security hardening |

## Where each contribution lives

- **Platform:** `app/main.py`, `app/portal.py`, `app/pharmacy.py`, `app/notify.py`, `app/history.py`, `app/templates/`, `app/static/`
- **Retrieval and seed data:** `handoff/`, `app/data/`
- **Router and integration:** `app/router.py`, `app/realist.py`, `app/ai.py`, `scripts/fit_router.py`
- **Evaluation:** `evaluation/README.md`, `evaluation/run_eval.py`, `evaluation/usage_bench.py`
