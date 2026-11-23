"""Real-world usage benchmark for MediPurse AI diagnosis (complements run_eval.py, which
measures accuracy).

Measures, on this machine:
  - cold start: time from launching the web app to the AI model being ready, and memory;
  - latency and throughput when several doctors submit cases at the same time (HTTP);
  - the LLM steps (DeepSeek): "Explain with AI" (4 calls), the realist's AI review and one
    history-agent turn: wall time, tokens, and actual spend (account balance before/after);
  - coverage of the knowledge behind the suggestions.

    python evaluation/usage_bench.py --clients 1,2,4,8 --per-client 12 --llm 4
Writes evaluation/usage.json. Starts its own app on port 8002 with a throw-away database.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run_eval import lat, pct  # noqa: E402

PORT = 8002
RATE: float | None = None   # INR per CNY, fetched in main()
BASE = f"http://localhost:{PORT}"


def _ps(cmd: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout


def mem_mb() -> dict:
    """Working set and peak of the process listening on PORT (the server itself, not the
    launcher that started it). Windows, via PowerShell."""
    try:
        out = _ps(f"$p=Get-Process -Id (Get-NetTCPConnection -LocalPort {PORT} -State Listen)[0].OwningProcess; "
                  "\"$($p.WorkingSet64) $($p.PeakWorkingSet64)\"").split()
        return {"working_set_mb": round(int(out[0]) / 2**20), "peak_mb": round(int(out[1]) / 2**20)}
    except Exception:
        return {}


def stop_app(proc):
    _ps(f"Get-NetTCPConnection -LocalPort {PORT} -State Listen -ErrorAction SilentlyContinue | "
        "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }")
    proc.terminate()
    proc.wait(timeout=30)


def start_app(tmp: Path, llm: bool = True) -> tuple[subprocess.Popen, float]:
    env = dict(os.environ, HMS_DB=str(tmp / "bench.db"), HMS_UPLOADS=str(tmp / "up"), SMTP_HOST="")
    if not llm:
        env["DEEPSEEK_API_KEY"] = ""   # an empty value wins over .env: AI steps switch to rule mode
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while time.perf_counter() - t0 < 180:
        try:
            if requests.get(BASE + "/api/ai-status", timeout=2).json().get("status") == "ready":
                return proc, time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("app did not become ready")


def load_test(cases: list[str], clients: int, per_client: int) -> dict:
    lock, times, errors = threading.Lock(), [], [0]

    def worker(k: int):
        s = requests.Session()
        s.post(BASE + "/login", data={"email": "dr.mehta@medipurse.test", "password": "demo1234"})
        for j in range(per_client):
            text = cases[(k * per_client + j) % len(cases)]
            t0 = time.perf_counter()
            r = s.post(BASE + "/patients/1/ai/new", data={"case_text": text, "symptoms": "[]"})
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                if r.status_code == 200 and "/ai/" in r.url:
                    times.append(dt)
                else:
                    errors[0] += 1

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(clients)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    return {"clients": clients, "requests": len(times), "errors": errors[0],
            "throughput_per_s": round(len(times) / wall, 2), **{k: v for k, v in lat(times).items() if k != "n"}}


def inr_per_cny() -> float | None:
    """Exchange rate for reporting spend in rupees (the DeepSeek account is billed in CNY)."""
    try:
        return float(requests.get("https://open.er-api.com/v6/latest/CNY", timeout=20).json()["rates"]["INR"])
    except Exception:
        return None


def balance_cny() -> float | None:
    try:
        from app.ai import AI
        llm = AI.llm()
        r = requests.get(llm.base.replace("/v1", "") + "/user/balance",
                         headers={"Authorization": f"Bearer {llm.api_key}"}, timeout=20).json()
        return float(next(b["total_balance"] for b in r["balance_infos"] if b["currency"] == "CNY"))
    except Exception:
        return None


def llm_bench(cases: list[dict], n: int) -> dict:
    import app.main  # noqa: F401  (loads .env: DeepSeek key)
    from app import history as hx
    from app import realist
    from app.ai import AI
    from types import SimpleNamespace
    AI._load()
    if not AI.llm_available(force=True):
        return {"skipped": "no LLM credit"}
    b0 = balance_cny()
    ex = []
    for c in cases[:n]:
        t0 = time.perf_counter()
        out = AI.explain(c["text"], case_id=c["id"])
        u = out["usage"]
        shown = sum(1 for d in out["diseases"].values() for cl in d["claims"] if cl.get("show"))
        total = sum(len(d["claims"]) for d in out["diseases"].values())
        ex.append({"id": c["id"], "s": time.perf_counter() - t0, "calls": u["calls"], "in": u["prompt_tokens"],
                   "out": u["completion_tokens"], "findings": len(out["findings"]), "claims": total, "claims_shown": shown})
    rv = []
    for c in cases[:max(n, 6)]:
        ranking = AI.rank(c["text"], k=10)
        rules = realist.assess_rules(c["text"], ranking)
        t0 = time.perf_counter()
        res = realist.assess_llm(AI.llm(), c["text"], [x["name"] for x in ranking["candidates"]], rules)
        rv.append({"s": time.perf_counter() - t0, "agree": res["verdict"] == rules["verdict"], "source": res["source"]})
    patient = SimpleNamespace(first_name="Asha", age=34, sex="Female", allergies="", full_name="Asha K")
    msgs, ht = [], []
    for said in ["I have had a cough and fever for four days", "It started on Sunday and is getting worse", "No, I do not smoke"]:
        if msgs:
            msgs.append({"role": "user", "text": said, "at": ""})
        t0 = time.perf_counter()
        turn = hx._ai_turn(AI.llm(), patient, msgs)
        ht.append(time.perf_counter() - t0)
        msgs.append({"role": "assistant", "text": turn.get("reply", ""), "at": ""})
    b1 = balance_cny()
    spend = round(b0 - b1, 4) if b0 is not None and b1 is not None else None
    return {
        "model": AI.llm().model,
        "explain": {"cases": len(ex), "p50_s": round(statistics.median(e["s"] for e in ex), 1),
                    "max_s": round(max(e["s"] for e in ex), 1), "calls_per_case": round(statistics.mean(e["calls"] for e in ex), 1),
                    "tokens_in_per_case": round(statistics.mean(e["in"] for e in ex)),
                    "tokens_out_per_case": round(statistics.mean(e["out"] for e in ex)),
                    "claims_per_case": round(statistics.mean(e["claims"] for e in ex), 1),
                    "claims_kept_after_checks": round(sum(e["claims_shown"] for e in ex) / max(1, sum(e["claims"] for e in ex)), 3),
                    "rows": ex},
        "realist_ai_review": {"cases": len(rv), "p50_s": round(statistics.median(x["s"] for x in rv), 1),
                              "max_s": round(max(x["s"] for x in rv), 1),
                              "agrees_with_router": round(sum(x["agree"] for x in rv) / len(rv), 3)},
        "history_ai_turn": {"turns": len(ht), "p50_s": round(statistics.median(ht), 1), "max_s": round(max(ht), 1)},
        "spend_cny_total": spend,
        "spend_inr_total": round(spend * RATE, 2) if spend is not None and RATE else None,
        "spend_note": "Actual account balance before minus after for everything in this section (billed in CNY, shown in INR).",
    }


def coverage() -> dict:
    d = json.loads((ROOT / "app/data/disease_drugs.json").read_text(encoding="utf-8"))
    cat = json.loads((ROOT / "handoff/weights/retriever/diseases.json").read_text(encoding="utf-8"))["diseases"]
    kg = json.loads((ROOT / "handoff/weights/catalog/optimuskg_catalog.json").read_text(encoding="utf-8"))
    ph = json.loads((ROOT / "app/data/phenotypes.json").read_text(encoding="utf-8"))
    from app.clinical import COMMON_CONDITIONS, COMMON_SYMPTOMS
    from app.drug_suggest import COMMON_CONDITION_MEDS
    size = sum(f.stat().st_size for f in (ROOT / "handoff/weights/retriever").glob("*"))
    return {"rare_disease_catalog": len(cat), "with_optimuskg_profile": len(kg),
            "with_case_derived_medicines": d["meta"]["with_case_drugs"],
            "with_approved_indications": d["meta"]["with_indications"],
            "searchable_phenotypes": ph["count"], "everyday_symptoms": len(COMMON_SYMPTOMS),
            "common_conditions": len(COMMON_CONDITIONS),
            "common_conditions_with_first_line_medicines": len(COMMON_CONDITION_MEDS),
            "retriever_weights_mb": round(size / 2**20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--per-client", type=int, default=12)
    ap.add_argument("--llm", type=int, default=4, help="cases for the LLM explanation benchmark (0 = skip)")
    a = ap.parse_args()
    global RATE
    RATE = inr_per_cny()
    rare = json.load(open(HERE / "synthetic_rare.json", encoding="utf-8"))["cases"]
    common = json.load(open(HERE / "synthetic_common.json", encoding="utf-8"))["cases"]
    texts = [c["text"] for pair in zip(rare, common) for c in pair]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "currency": {"inr_per_cny": RATE, "source": "open.er-api.com"}, "machine": "development laptop, Windows 11, CPU only, single uvicorn worker"}
    # 1) AI review off: the pure local path (retriever + router), as without DeepSeek credit
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proc, ready_s = start_app(Path(tmp), llm=False)
        try:
            out["cold_start"] = {"seconds_to_ready": round(ready_s, 1), **mem_mb()}
            for c in texts[:4]:   # warm up
                load_test([c], 1, 1)
            out["load_local"] = [load_test(texts, int(c), a.per_client) for c in a.clients.split(",")]
            out["memory_after_load"] = mem_mb()
        finally:
            stop_app(proc)
    # 2) AI review on: each analysis also starts the realist's DeepSeek review in the background
    if a.llm:
        import app.main  # noqa: F401  (loads .env)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            proc, _ = start_app(Path(tmp), llm=True)
            try:
                b0 = balance_cny()
                for c in texts[:2]:
                    load_test([c], 1, 1)
                out["load_with_ai_review"] = [load_test(texts, c, 8) for c in (1, 4)]
                time.sleep(8)   # let the last background reviews finish before reading the balance
                b1 = balance_cny()
                n = 2 + sum(x["requests"] for x in out["load_with_ai_review"])
                cny = round(b0 - b1, 4) if b0 and b1 else None
                out["ai_review_spend"] = {"analyses": n, "spend_cny": cny,
                                          "spend_inr": round(cny * RATE, 2) if cny is not None and RATE else None}
            finally:
                stop_app(proc)
    out["coverage"] = coverage()
    if a.llm:
        mixed = [c for pair in zip(rare[:a.llm], common[:a.llm]) for c in pair][:a.llm]
        out["llm"] = llm_bench(mixed, a.llm)
    (HERE / "usage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "llm"}, indent=1))
    if "llm" in out:
        print(json.dumps({k: v for k, v in out["llm"].items() if k != "explain"} | {"explain": {k: v for k, v in out["llm"].get("explain", {}).items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
