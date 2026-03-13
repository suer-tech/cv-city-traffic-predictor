from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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

snapshot_root_path = Path(settings.snapshot_root)
snapshot_root_path.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(snapshot_root_path)), name="snapshots")


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


@app.get("/debug/annotated")
def list_annotated(camera_id: str | None = None, limit: int = Query(default=10, ge=1, le=50)) -> list[dict]:
    root = snapshot_root_path / "annotated"
    pattern = "*/*_annotated.png" if camera_id is None else f"{camera_id}/*_annotated.png"
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    results: list[dict] = []
    for p in files:
        rel = p.relative_to(snapshot_root_path)
        camera = p.parent.name
        results.append(
            {
                "camera_id": camera,
                "file_path": str(p),
                "url": f"/snapshots/{rel.as_posix()}",
                "modified_at_utc": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(),
            }
        )
    return results


@app.post("/jobs/run-once", response_model=JobRunResponse)
def run_jobs_once() -> JobRunResponse:
    annotated = run_pipeline_once(save_debug_artifacts=True, force_real_capture=True)
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow(), annotated_images=annotated)


@app.post("/jobs/capture", response_model=JobRunResponse)
def run_capture() -> JobRunResponse:
    annotated = capture_and_extract_job(save_debug_artifacts=True, force_real_capture=True)
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow(), annotated_images=annotated)


@app.post("/jobs/inference", response_model=JobRunResponse)
def run_inference() -> JobRunResponse:
    inference_job()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow(), annotated_images=[])


@app.post("/jobs/retrain", response_model=JobRunResponse)
def run_retrain() -> JobRunResponse:
    retrain_job()
    return JobRunResponse(status="ok", started_at_utc=datetime.utcnow(), annotated_images=[])


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
    input { width: 100%; margin: 4px 0; }
    #annotated img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>CV City Traffic Predictor Admin UI</h1>
  <div class='card'>
    <button onclick='runJob("/jobs/run-once")'>Run full pipeline once</button>
    <button onclick='runJob("/jobs/capture")'>Run capture+detect once</button>
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

  <div class='card'><h3>Latest annotated detection</h3><div id='annotated'>No annotated images yet.</div></div>
  <div class='card'><h3>Cameras</h3><pre id='cameras'></pre></div>
  <div class='card'><h3>Routes</h3><pre id='routes'></pre></div>
  <div class='card'><h3>Predictions</h3><pre id='preds'></pre></div>

<script>
async function j(url, opts={}) { const r = await fetch(url, {headers:{'Content-Type':'application/json'}, ...opts}); return r.json(); }
async function runJob(url){ const res = await j(url, {method:'POST'}); if(res.annotated_images && res.annotated_images.length){ console.log('annotated', res.annotated_images[0]); } await loadData(); }
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
function renderAnnotated(items){
  if(!items.length){ annotated.innerHTML = 'No annotated images yet.'; return; }
  const x = items[0];
  annotated.innerHTML = `<div><b>${x.camera_id}</b> @ ${x.modified_at_utc}</div><img src='${x.url}?t=${Date.now()}' alt='annotated detection'>`;
}
async function loadData(){
  const [s,c,r,p,a] = await Promise.all([
    j('/dashboard/summary'), j('/cameras'), j('/routes'), j('/predictions?limit=30'), j('/debug/annotated?limit=5')
  ]);
  summary.textContent = 'Cameras: '+s.enabled_cameras+' | Routes: '+s.enabled_routes+' | Predictions: '+s.last_predictions;
  cameras.textContent = JSON.stringify(c, null, 2);
  routes.textContent = JSON.stringify(r, null, 2);
  preds.textContent = JSON.stringify(p, null, 2);
  renderAnnotated(a);
}
loadData();
</script>
</body></html>
"""
    return HTMLResponse(content=html)
