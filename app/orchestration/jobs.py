import logging
from datetime import datetime, timedelta
from pathlib import Path

from app.capture.service import CaptureService
from app.config import settings
from app.evaluation.service import EvaluationService
from app.features.service import floor_to_5m
from app.forecast.service import ForecastService
from app.storage.db import SessionLocal
from app.storage.models import TrafficMetric5m, TrafficPrediction
from app.storage.repository import Repository
from app.vision.service import VisionService

logger = logging.getLogger(__name__)

capture_service = CaptureService()
vision_service = VisionService()
forecast_service = ForecastService()
evaluation_service = EvaluationService()


def _annotated_output_path(camera_id: str, ts: datetime) -> Path:
    filename = ts.strftime("%Y%m%d_%H%M%S") + "_annotated.png"
    return Path(settings.snapshot_root) / "annotated" / camera_id / filename


def capture_and_extract_job(save_debug_artifacts: bool = False, force_real_capture: bool = False) -> list[str]:
    now = datetime.utcnow()
    annotated_paths: list[str] = []
    with SessionLocal() as session:
        repo = Repository(session)
        cameras = repo.list_enabled_cameras()
        for camera in cameras:
            capture = capture_service.capture_with_retries(
                camera.camera_id,
                camera.source_url,
                force_real=force_real_capture,
            )
            if capture is None:
                continue

            vision = vision_service.analyze(capture.image_path)
            source_image_uri = capture.image_path
            if save_debug_artifacts:
                annotated_path = vision_service.save_annotated(
                    capture.image_path,
                    vision.boxes,
                    str(_annotated_output_path(camera.camera_id, capture.capture_ts_utc)),
                )
                source_image_uri = annotated_path
                annotated_paths.append(annotated_path)

            metric = TrafficMetric5m(
                timestamp_utc=floor_to_5m(capture.capture_ts_utc),
                camera_id=camera.camera_id,
                intersection_id=camera.intersection_id,
                lane_id="all",
                vehicle_count=vision.vehicle_count,
                density=vision.density,
                avg_speed_proxy=None,
                quality_score=vision.quality_score,
                source_image_uri=source_image_uri,
            )
            repo.upsert_metric(metric)
        session.commit()
    logger.info(
        "capture_and_extract_job_completed",
        extra={"ts": now.isoformat(), "save_debug_artifacts": save_debug_artifacts},
    )
    return annotated_paths


def retrain_job() -> None:
    with SessionLocal() as session:
        repo = Repository(session)
        for camera in repo.list_enabled_cameras():
            metrics = repo.list_recent_metrics(camera.camera_id, limit=4000)
            history = [(m.timestamp_utc, float(m.vehicle_count)) for m in metrics]
            forecast_service.train(history)
    logger.info("retrain_job_completed")


def inference_job() -> None:
    now = datetime.utcnow().replace(second=0, microsecond=0)
    with SessionLocal() as session:
        repo = Repository(session)
        for camera in repo.list_enabled_cameras():
            metrics = repo.list_recent_metrics(camera.camera_id, limit=4000)
            history = [(m.timestamp_utc, float(m.vehicle_count)) for m in metrics]
            preds = forecast_service.predict(history, now)
            for horizon, pred in preds.items():
                prediction = TrafficPrediction(
                    prediction_ts_utc=now,
                    target_ts_utc=now + timedelta(minutes=horizon),
                    horizon_min=horizon,
                    entity_type="camera",
                    entity_id=camera.camera_id,
                    pred_vehicle_count=pred,
                    pred_density=min(pred / 80.0, 1.0),
                    congestion_probability=forecast_service.congestion_probability(
                        pred, settings.camera_congestion_threshold_count
                    ),
                    model_version="baseline_bias_v2",
                    features_version="fv_v2",
                    explainability_payload={"top_features": ["lag_1", "roll_3", "hour"]},
                )
                repo.upsert_prediction(prediction)

            ext = evaluation_service.pull_external_traffic("camera", camera.camera_id)
            repo.add_external_observation(evaluation_service.as_model(now, "camera", camera.camera_id, ext))

        for route in repo.list_enabled_routes():
            camera_ids = repo.list_route_camera_ids(route.route_id)
            route_count = float(repo.aggregate_route_latest_count(camera_ids))
            for horizon in (10, 20, 30):
                pred = route_count
                prediction = TrafficPrediction(
                    prediction_ts_utc=now,
                    target_ts_utc=now + timedelta(minutes=horizon),
                    horizon_min=horizon,
                    entity_type="route",
                    entity_id=route.route_id,
                    pred_vehicle_count=pred,
                    pred_density=min(pred / 120.0, 1.0),
                    congestion_probability=forecast_service.congestion_probability(
                        pred, settings.route_congestion_threshold_count
                    ),
                    model_version="route_naive_v2",
                    features_version="route_fv_v2",
                    explainability_payload={"top_features": ["latest_route_count"]},
                )
                repo.upsert_prediction(prediction)

            ext = evaluation_service.pull_external_traffic("route", route.route_id)
            repo.add_external_observation(evaluation_service.as_model(now, "route", route.route_id, ext))

        session.commit()
    logger.info("inference_job_completed", extra={"ts": now.isoformat()})


def run_pipeline_once(save_debug_artifacts: bool = True, force_real_capture: bool = False) -> list[str]:
    annotated = capture_and_extract_job(
        save_debug_artifacts=save_debug_artifacts,
        force_real_capture=force_real_capture,
    )
    inference_job()
    return annotated
