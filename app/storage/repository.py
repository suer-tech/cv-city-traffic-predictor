from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Select, desc, func, select
from sqlalchemy.orm import Session

from app.storage.models import (
    Camera,
    ExternalTrafficObservation,
    Route,
    RouteCamera,
    TrafficMetric5m,
    TrafficPrediction,
)


class Repository:
    def __init__(self, session: Session):
        self.session = session

    def list_enabled_cameras(self) -> Sequence[Camera]:
        stmt: Select[tuple[Camera]] = select(Camera).where(Camera.is_enabled.is_(True)).order_by(Camera.camera_id)
        return self.session.scalars(stmt).all()

    def list_cameras(self) -> Sequence[Camera]:
        return self.session.scalars(select(Camera).order_by(Camera.camera_id)).all()

    def get_camera(self, camera_id: str) -> Camera | None:
        return self.session.get(Camera, camera_id)

    def upsert_camera(self, camera: Camera) -> Camera:
        existing = self.session.get(Camera, camera.camera_id)
        if existing:
            existing.source_url = camera.source_url
            existing.intersection_id = camera.intersection_id
            existing.direction = camera.direction
            existing.is_enabled = camera.is_enabled
            existing.roi_config = camera.roi_config
            existing.updated_at_utc = datetime.utcnow()
            self.session.add(existing)
            return existing
        self.session.add(camera)
        return camera

    def upsert_metric(self, metric: TrafficMetric5m) -> TrafficMetric5m:
        stmt = select(TrafficMetric5m).where(
            TrafficMetric5m.timestamp_utc == metric.timestamp_utc,
            TrafficMetric5m.camera_id == metric.camera_id,
        )
        existing = self.session.scalar(stmt)
        if existing:
            existing.intersection_id = metric.intersection_id
            existing.lane_id = metric.lane_id
            existing.vehicle_count = metric.vehicle_count
            existing.density = metric.density
            existing.avg_speed_proxy = metric.avg_speed_proxy
            existing.quality_score = metric.quality_score
            existing.source_image_uri = metric.source_image_uri
            self.session.add(existing)
            return existing
        self.session.add(metric)
        return metric

    def list_recent_metrics(self, camera_id: str, limit: int = 288) -> list[TrafficMetric5m]:
        stmt = (
            select(TrafficMetric5m)
            .where(TrafficMetric5m.camera_id == camera_id)
            .order_by(TrafficMetric5m.timestamp_utc.desc())
            .limit(limit)
        )
        return list(reversed(self.session.scalars(stmt).all()))

    def list_metrics_window(self, camera_id: str, limit: int = 200) -> list[TrafficMetric5m]:
        stmt = (
            select(TrafficMetric5m)
            .where(TrafficMetric5m.camera_id == camera_id)
            .order_by(desc(TrafficMetric5m.timestamp_utc))
            .limit(limit)
        )
        return self.session.scalars(stmt).all()

    def upsert_prediction(self, prediction: TrafficPrediction) -> TrafficPrediction:
        stmt = select(TrafficPrediction).where(
            TrafficPrediction.prediction_ts_utc == prediction.prediction_ts_utc,
            TrafficPrediction.target_ts_utc == prediction.target_ts_utc,
            TrafficPrediction.entity_type == prediction.entity_type,
            TrafficPrediction.entity_id == prediction.entity_id,
            TrafficPrediction.horizon_min == prediction.horizon_min,
        )
        existing = self.session.scalar(stmt)
        if existing:
            existing.pred_vehicle_count = prediction.pred_vehicle_count
            existing.pred_density = prediction.pred_density
            existing.congestion_probability = prediction.congestion_probability
            existing.model_version = prediction.model_version
            existing.features_version = prediction.features_version
            existing.explainability_payload = prediction.explainability_payload
            self.session.add(existing)
            return existing
        self.session.add(prediction)
        return prediction

    def list_predictions(self, entity_type: str | None = None, entity_id: str | None = None, limit: int = 100) -> list[TrafficPrediction]:
        stmt = select(TrafficPrediction)
        if entity_type:
            stmt = stmt.where(TrafficPrediction.entity_type == entity_type)
        if entity_id:
            stmt = stmt.where(TrafficPrediction.entity_id == entity_id)
        stmt = stmt.order_by(desc(TrafficPrediction.prediction_ts_utc)).limit(limit)
        return self.session.scalars(stmt).all()

    def list_enabled_routes(self) -> Sequence[Route]:
        return self.session.scalars(select(Route).where(Route.is_enabled.is_(True)).order_by(Route.route_id)).all()

    def list_routes(self) -> Sequence[Route]:
        return self.session.scalars(select(Route).order_by(Route.route_id)).all()

    def upsert_route(self, route: Route) -> Route:
        existing = self.session.get(Route, route.route_id)
        if existing:
            existing.name = route.name
            existing.is_enabled = route.is_enabled
            self.session.add(existing)
            return existing
        self.session.add(route)
        return route

    def replace_route_cameras(self, route_id: str, camera_ids: list[str]) -> None:
        self.session.query(RouteCamera).filter(RouteCamera.route_id == route_id).delete()
        for idx, camera_id in enumerate(camera_ids):
            self.session.add(RouteCamera(route_id=route_id, camera_id=camera_id, order_index=idx))

    def list_route_camera_ids(self, route_id: str) -> list[str]:
        stmt = (
            select(RouteCamera.camera_id)
            .where(RouteCamera.route_id == route_id)
            .order_by(RouteCamera.order_index.asc())
        )
        return list(self.session.scalars(stmt).all())

    def aggregate_route_latest_count(self, camera_ids: list[str]) -> int:
        if not camera_ids:
            return 0
        subq = (
            select(
                TrafficMetric5m.camera_id,
                func.max(TrafficMetric5m.timestamp_utc).label("max_ts"),
            )
            .where(TrafficMetric5m.camera_id.in_(camera_ids))
            .group_by(TrafficMetric5m.camera_id)
            .subquery()
        )
        stmt = (
            select(func.sum(TrafficMetric5m.vehicle_count))
            .join(
                subq,
                (TrafficMetric5m.camera_id == subq.c.camera_id)
                & (TrafficMetric5m.timestamp_utc == subq.c.max_ts),
            )
        )
        value = self.session.scalar(stmt)
        return int(value or 0)

    def add_external_observation(self, obs: ExternalTrafficObservation) -> None:
        self.session.add(obs)

    def latest_external_observation(self, entity_type: str, entity_id: str) -> ExternalTrafficObservation | None:
        stmt = (
            select(ExternalTrafficObservation)
            .where(
                ExternalTrafficObservation.entity_type == entity_type,
                ExternalTrafficObservation.entity_id == entity_id,
            )
            .order_by(desc(ExternalTrafficObservation.timestamp_utc))
            .limit(1)
        )
        return self.session.scalar(stmt)
