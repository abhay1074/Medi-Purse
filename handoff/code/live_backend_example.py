"""Minimal FastAPI skeleton for the live chat agent.

This is a starting point, not a production server. It shows the session model,
asynchronous case analysis, SSE progress, and report retrieval. Replace the
in-memory stores with Postgres/Redis and put the analysis job on a worker queue.

Run:
    uv pip install fastapi uvicorn
    uvicorn live_backend_example:app --port 8000
"""

from __future__ import annotations

import json
import queue
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from live_engine import LiveEngine

app = FastAPI(title="Glass-box rare disease agent")
engine = LiveEngine("../", llm_base="http://127.0.0.1:8080/v1")

SESSIONS: dict[str, dict] = {}
REPORTS: dict[str, dict] = {}
JOBS: dict[str, "queue.Queue"] = {}


class NewSession(BaseModel):
    model_id: str | None = None


class Message(BaseModel):
    type: str              # "analyze_case" | "follow_up"
    text: str
    images: list[dict] = []


@app.post("/v1/sessions")
def create_session(body: NewSession):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"id": sid, "model_id": body.model_id,
                     "turns": [], "report_id": None}
    return {"session_id": sid}


@app.post("/v1/sessions/{sid}/messages")
def post_message(sid: str, msg: Message):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")

    if msg.type == "follow_up":
        rid = s.get("report_id")
        if not rid:
            raise HTTPException(409, "no case analysis in this session yet")
        s["turns"].append({"role": "user", "type": "follow_up", "text": msg.text})
        report = REPORTS[rid]
        needle = msg.text.lower()
        hits = [c for c in report["claims"]
                if needle in c["claim"].lower() or needle in c["disease"].lower()]
        answer = ("Here are the stored, cited claims that match your question."
                  if hits else
                  "I can answer from the stored report. Try naming a disease or a claim.")
        s["turns"].append({"role": "assistant", "text": answer})
        return {"answer": answer, "claims": hits[:10],
                "final_order": report["final_order"]}

    if msg.type != "analyze_case":
        raise HTTPException(400, "type must be analyze_case or follow_up")

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = q
    rid = str(uuid.uuid4())

    def run():
        def progress(stage, pct):
            q.put({"event": "stage", "data": {"stage": stage, "progress": pct}})
        try:
            report = engine.analyze(msg.text, case_id=sid, images=msg.images,
                                    progress=progress)
            REPORTS[rid] = report
            s["report_id"] = rid
            s["turns"].append({"role": "user", "type": "analyze_case", "text": msg.text})
            s["turns"].append({"role": "assistant", "type": "report", "report_id": rid})
            q.put({"event": "done", "data": {"report_id": rid}})
        except Exception as exc:  # keep the SSE stream alive
            q.put({"event": "error", "data": {"message": str(exc)}})
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "report_id": rid}


@app.get("/v1/jobs/{job_id}/events")
def job_events(job_id: str):
    q = JOBS.get(job_id)
    if not q:
        raise HTTPException(404, "unknown job")

    def stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/reports/{rid}")
def get_report(rid: str):
    if rid not in REPORTS:
        raise HTTPException(404, "unknown report")
    return REPORTS[rid]


@app.get("/v1/sessions/{sid}")
def get_session(sid: str):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")
    return s
