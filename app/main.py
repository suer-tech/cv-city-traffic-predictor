from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.admin_ui.schemas import (
    CameraToggleRequest,
    CameraUpsertRequest,
    DashboardSummary,
    JobRunResponse,
    RoiUpdateRequest,
    RouteUpsertRequest,
)
from app.config import settings
from app.logging import configure_logging
from app.orchestration.jobs import capture_and_extract_job, inference_job, retrain_job, run_pipeline_once
from app.orchestration.scheduler import start_scheduler
from app.storage.db import Base, SessionLocal, engine
from app.storage.models import Camera, Route
from app.storage.repository import Repository

configure_logging()
app = FastAPI(title=settings.ui_title)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    _seed_defaults()
    if settings.scheduler_enabled:
        start_scheduler()


def _seed_defaults() -> None:
    with SessionLocal() as session:
        repo = Repository(session)
        if repo.get_camera(settings.default_camera_id) is None:
            repo.upsert_camera(
                Camera(
                    camera_id=settings.default_camera_id,
                    source_url=settings.default_camera_url,
                    intersection_id=settings.default_intersection_id,
                    direction=settings.default_direction,
                    is_enabled=True,
                    roi_config={},
                )
            )
        if session.get(Route, "route_default") is None:
            repo.upsert_route(Route(route_id="route_default", name="Default route", is_enabled=True))
            repo.replace_route_cameras("route_default", [settings.default_camera_id])
        session.commit()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary() -> DashboardSummary:
    with SessionLocal() as session:
        repo = Repository(session)
        return DashboardSummary(
            enabled_cameras=len(repo.list_enabled_cameras()),
            enabled_routes=len(repo.list_enabled_routes()),
            last_predictions=len(repo.list_predictions(limit=20)),
            generated_at_utc=datetime.utcnow(),
        )


@app.get("/cameras")
def list_cameras() -> list[dict]:
    with SessionLocal() as session:
        repo = Repository(session)
        cameras = repo.list_cameras()
        return [
            {
                "camera_id": c.camera_id,
                "source_url": c.source_url,
                "intersection_id": c.intersection_id,
                "direction": c.direction,
                "is_enabled": c.is_enabled,
                "roi_config": c.roi_config,
            }
            for c in cameras
        ]


@app.post("/cameras")
def upsert_camera(payload: CameraUpsertRequest) -> dict:
    with SessionLocal() as session:
        repo = Repository(session)
        camera = Camera(**payload.model_dump())
        repo.upsert_camera(camera)
        session.commit()
    return {"status": "ok", "camera_id": payload.camera_id}


@app.patch("/cameras/{camera_id}/toggle")
def toggle_camera(camera_id: str, payload: CameraToggleRequest) -> dict:
    with SessionLocal() as session:
        camera = session.get(Camera, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera.is_enabled = payload.is_enabled
        session.add(camera)
        session.commit()
    return {"status": "ok", "camera_id": camera_id, "is_enabled": payload.is_enabled}


@app.patch("/cameras/{camera_id}/roi")
def update_roi(camera_id: str, payload: RoiUpdateRequest) -> dict:
    with SessionLocal() as session:
        camera = session.get(Camera, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera.roi_config = payload.roi_config
        session.add(camera)
        session.commit()
    return {"status": "ok", "camera_id": camera_id, "roi_config": payload.roi_config}


@app.get("/metrics")
def list_metrics(camera_id: str, limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    with SessionLocal() as session:
        repo = Repository(session)
        metrics = repo.list_metrics_window(camera_id=camera_id, limit=limit)
        return [
            {
                "timestamp_utc": m.timestamp_utc.isoformat(),
                "camera_id": m.camera_id,
                "vehicle_count": m.vehicle_count,
                "density": m.density,
                "quality_score": m.quality_score,
                "source_image_uri": m.source_image_uri,
            }
            for m in metrics
        ]


@app.get("/predictions")
def list_predictions(
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    with SessionLocal() as session:
        repo = Repository(session)
        predictions = repo.list_predictions(entity_type=entity_type, entity_id=entity_id, limit=limit)
        return [
            {
                "prediction_ts_utc": p.prediction_ts_utc.isoformat(),
                "target_ts_utc": p.target_ts_utc.isoformat(),
                "horizon_min": p.horizon_min,
                "entity_type": p.entity_type,
                "entity_id": p.entity_id,
                "pred_vehicle_count": p.pred_vehicle_count,
                "pred_density": p.pred_density,
                "congestion_probability": p.congestion_probability,
                "explainability_payload": p.explainability_payload,
            }
            for p in predictions
        ]


@app.get("/routes")
def list_routes() -> list[dict]:
    with SessionLocal() as session:
        repo = Repository(session)
        routes = repo.list_routes()
        return [
            {
                "route_id": r.route_id,
                "name": r.name,
                "is_enabled": r.is_enabled,
                "camera_ids": repo.list_route_camera_ids(r.route_id),
            }
            for r in routes
        ]


@app.post("/routes")
def upsert_route(payload: RouteUpsertRequest) -> dict:
    with SessionLocal() as session:
        repo = Repository(session)
        route = Route(route_id=payload.route_id, name=payload.name, is_enabled=payload.is_enabled)
        repo.upsert_route(route)
        repo.replace_route_cameras(payload.route_id, payload.camera_ids)
        session.commit()
    return {"status": "ok", "route_id": payload.route_id}


@app.post("/jobs/run-once", response_model=JobRunResponse)
def run_jobs_once() -> JobRunResponse:
    run_pipeline_once()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow())


@app.post("/jobs/capture", response_model=JobRunResponse)
def run_capture() -> JobRunResponse:
    capture_and_extract_job()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow())


@app.post("/jobs/inference", response_model=JobRunResponse)
def run_inference() -> JobRunResponse:
    inference_job()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow())


@app.post("/jobs/retrain", response_model=JobRunResponse)
def run_retrain() -> JobRunResponse:
    retrain_job()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow())


@app.get("/", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    html = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Traffic Admin</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f7f8fa; }
    .card { background: white; border-radius: 10px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    button { margin-right: 8px; }
    pre { max-height: 260px; overflow:auto; background: #101522; color:#e9efff; padding: 10px; border-radius:8px; }
    input, textarea { width: 100%; margin: 4px 0; }
  </style>
</head>
<body>
  <h1>CV City Traffic Predictor Admin UI</h1>
  <div class='card'>
    <button onclick='runJob("/jobs/run-once")'>Run full pipeline once</button>
    <button onclick='loadData()'>Refresh</button>
    <div id='summary'></div>
  </div>

  <div class='card'>
    <h3>Add/Update Camera</h3>
    <input id='camera_id' placeholder='camera_id'>
    <input id='source_url' placeholder='source_url'>
    <input id='intersection_id' placeholder='intersection_id'>
    <input id='direction' value='SN' placeholder='direction'>
    <button onclick='saveCamera()'>Save camera</button>
  </div>

  <div class='card'>
    <h3>Configure Route</h3>
    <input id='route_id' placeholder='route_id'>
    <input id='route_name' placeholder='route_name'>
    <input id='route_cameras' placeholder='camera ids comma separated'>
    <button onclick='saveRoute()'>Save route</button>
  </div>

  <div class='card'><h3>Cameras</h3><pre id='cameras'></pre></div>
  <div class='card'><h3>Routes</h3><pre id='routes'></pre></div>
  <div class='card'><h3>Predictions</h3><pre id='preds'></pre></div>

<script>
async function j(url, opts={}) { const r = await fetch(url, {headers:{'Content-Type':'application/json'}, ...opts}); return r.json(); }
async function runJob(url){ await j(url, {method:'POST'}); await loadData(); }
async function saveCamera(){
  const payload = {
    camera_id: camera_id.value, source_url: source_url.value, intersection_id: intersection_id.value,
    direction: direction.value || 'SN', is_enabled: true, roi_config: {}
  };
  await j('/cameras', {method:'POST', body: JSON.stringify(payload)});
  await loadData();
}
async function saveRoute(){
  const payload = {
    route_id: route_id.value, name: route_name.value, is_enabled: true,
    camera_ids: route_cameras.value.split(',').map(v=>v.trim()).filter(Boolean)
  };
  await j('/routes', {method:'POST', body: JSON.stringify(payload)});
  await loadData();
}
async function loadData(){
  const [s,c,r,p] = await Promise.all([j('/dashboard/summary'), j('/cameras'), j('/routes'), j('/predictions?limit=30')]);
  summary.textContent = 'Cameras: '+s.enabled_cameras+' | Routes: '+s.enabled_routes+' | Predictions: '+s.last_predictions;
  cameras.textContent = JSON.stringify(c, null, 2);
  routes.textContent = JSON.stringify(r, null, 2);
  preds.textContent = JSON.stringify(p, null, 2);
}
loadData();
</script>
</body></html>
"""
    return HTMLResponse(content=html)
