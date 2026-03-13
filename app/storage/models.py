from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_url: Mapped[str] = mapped_column(String(512), nullable=False)
    intersection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(32), nullable=False, default="SN")
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    roi_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)
    updated_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class TrafficMetric5m(Base):
    __tablename__ = "traffic_metrics_5m"
    __table_args__ = (UniqueConstraint("timestamp_utc", "camera_id", name="uq_metric_ts_camera"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"), nullable=False)
    intersection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    lane_id: Mapped[str] = mapped_column(String(64), nullable=False, default="all")
    vehicle_count: Mapped[int] = mapped_column(Integer, nullable=False)
    density: Mapped[float] = mapped_column(Float, nullable=False)
    avg_speed_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    source_image_uri: Mapped[str] = mapped_column(String(512), nullable=False)


class Route(Base):
    __tablename__ = "routes"

    route_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RouteCamera(Base):
    __tablename__ = "route_cameras"
    __table_args__ = (UniqueConstraint("route_id", "camera_id", name="uq_route_camera"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.route_id"), nullable=False)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)


class TrafficPrediction(Base):
    __tablename__ = "traffic_predictions"
    __table_args__ = (
        UniqueConstraint("prediction_ts_utc", "target_ts_utc", "horizon_min", "entity_type", "entity_id", name="uq_prediction_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    target_ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    horizon_min: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(16), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    pred_vehicle_count: Mapped[float] = mapped_column(Float, nullable=False)
    pred_density: Mapped[float] = mapped_column(Float, nullable=False)
    congestion_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    features_version: Mapped[str] = mapped_column(String(64), nullable=False)
    explainability_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class ExternalTrafficObservation(Base):
    __tablename__ = "external_traffic_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(16), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="yandex")
    traffic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_status: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
