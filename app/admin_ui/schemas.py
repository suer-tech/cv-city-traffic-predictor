from datetime import datetime

from pydantic import BaseModel, Field


class CameraUpsertRequest(BaseModel):
    camera_id: str
    source_url: str
    intersection_id: str
    direction: str = "SN"
    is_enabled: bool = True
    roi_config: dict = Field(default_factory=dict)


class CameraToggleRequest(BaseModel):
    is_enabled: bool


class RoiUpdateRequest(BaseModel):
    roi_config: dict


class RouteUpsertRequest(BaseModel):
    route_id: str
    name: str
    is_enabled: bool = True
    camera_ids: list[str] = Field(default_factory=list)


class JobRunResponse(BaseModel):
    status: str
    started_at_utc: datetime


class DashboardSummary(BaseModel):
    enabled_cameras: int
    enabled_routes: int
    last_predictions: int
    generated_at_utc: datetime
