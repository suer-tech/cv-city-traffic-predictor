from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://traffic:traffic@localhost:5432/traffic"
    mlflow_tracking_uri: str = "http://localhost:5000"

    capture_interval_minutes: int = 5
    inference_interval_minutes: int = 10
    retrain_interval_hours: int = 24
    capture_retry_count: int = 3
    capture_provider: str = "synthetic"
    capture_timeout_ms: int = 20_000

    camera_congestion_threshold_count: int = 25
    route_congestion_threshold_count: int = 60

    cv_count_mape_threshold: float = 0.35
    forecast_mape_10_threshold: float = 0.30
    forecast_mape_20_threshold: float = 0.35
    forecast_mape_30_threshold: float = 0.40

    snapshot_root: str = "data/snapshots"
    persist_snapshot_every_minutes: int = 60

    scheduler_enabled: bool = True
    ui_title: str = "CV City Traffic Predictor"

    default_camera_id: str = "1479988588"
    default_camera_url: str = "http://maps.ufanet.ru/ufa#1479988588"
    default_intersection_id: str = "ufa_default_intersection"
    default_direction: str = "SN"

    yandex_provider_enabled: bool = False
    yandex_provider_name: str = "yandex"
    yandex_score_default: float | None = Field(default=None)


settings = Settings()
