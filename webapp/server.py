"""Web server for the live lab (FastAPI).

    python -m webapp            # then open http://127.0.0.1:8000

Endpoints
    GET  /                          the single-page app (static/index.html)
    GET  /api/defaults              default settings and allowed ranges
    POST /api/runs                  start a training run with the posted settings
    GET  /api/runs                  all runs with their final metrics (runs table)
    GET  /api/runs/{id}/events      live progress as server-sent events
    POST /api/runs/{id}/predict     explain one patient: {"x": [8 values]}
    GET  /api/runs/{id}/model       the current (possibly edited) rule base and its metrics
    POST /api/runs/{id}/rules       replace the rule base: {"antecedents": [[...], ...]}
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import jobs

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Glass-Box Insight lab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
store = jobs.JobStore()


def _job(run_id: str, ready: bool = False) -> jobs.Job:
    job = store.get(run_id)
    if job is None:
        raise HTTPException(404, f"no run {run_id}")
    if ready and job.summary is None:
        raise HTTPException(409, "the run has not finished training yet")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/defaults")
def defaults():
    return {"defaults": jobs.DEFAULTS, "options": jobs.OPTIONS}


@app.post("/api/runs")
async def start_run(request: Request):
    try:
        job = store.start(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    return {"id": job.id, "config": job.cfg}


@app.get("/api/runs")
def list_runs():
    return [jobs.run_row(job) for job in store.all()]


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str, request: Request):
    job = _job(run_id)
    start = int(request.headers.get("last-event-id", -1)) + 1

    def stream():
        i = start
        while True:
            batch = job.wait_events(i)
            for event in batch:
                yield f"id: {event['id']}\ndata: {json.dumps(event)}\n\n"
            i += len(batch)
            if job.finished and i >= len(job.events):
                yield "event: end\ndata: {}\n\n"
                return
            if not batch:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/runs/{run_id}/predict")
async def predict(run_id: str, request: Request):
    job = _job(run_id, ready=True)
    try:
        return jobs.predict(job, (await request.json())["x"])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/runs/{run_id}/model")
def current_model(run_id: str):
    job = _job(run_id, ready=True)
    return {"model": jobs.describe_model(job.model), "metrics": job.evaluate(job.model),
            "trained": job.summary["final"]}


@app.post("/api/runs/{run_id}/rules")
async def edit_rules(run_id: str, request: Request):
    job = _job(run_id, ready=True)
    try:
        return jobs.edit_rules(job, (await request.json())["antecedents"])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
