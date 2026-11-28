# Contributors

MediPurse is a three-person team project. The commit history is maintained by Abhay Yadav, so it
does not reflect how the work was split.

| Contributor | Areas |
|---|---|
| **Animesh Pathak** | Rare-disease retrieval and explanation engine (Glassbox fusion v4 SHAP-8), built with Aarav Singh; realist router; backend and backend–frontend–LLM integration, with Abhay Yadav; evaluation; security hardening |
| **Aarav Singh** | Rare-disease retrieval and explanation engine (Glassbox fusion v4 SHAP-8), built with Animesh Pathak; the `handoff/` bundle is committed through Aarav's account |
| **Abhay Yadav** | Frontend and clinic platform; backend and backend–frontend–LLM integration, with Animesh Pathak; repository maintainer |

## How the work was split

- **Animesh Pathak and Aarav Singh** built the original Glassbox retriever and explanation engine.
  The `handoff/` bundle comes from that work and was committed through Aarav's account.
- **Abhay Yadav** built the frontend and the clinic platform around it.
- **Animesh Pathak and Abhay Yadav** then built the backend and the backend–frontend–LLM glue.
- **Animesh Pathak** built the realist router and the evaluation, and did the security hardening.

## Where each contribution lives

- **Retrieval and explanation engine:** `handoff/`
- **Router:** `app/router.py`, `scripts/fit_router.py`
- **Backend and integration:** `app/main.py`, `app/ai.py`, `app/realist.py`
- **Frontend and platform:** `app/portal.py`, `app/pharmacy.py`, `app/notify.py`, `app/history.py`, `app/templates/`, `app/static/`
- **Evaluation:** `evaluation/README.md`, `evaluation/run_eval.py`, `evaluation/usage_bench.py`
