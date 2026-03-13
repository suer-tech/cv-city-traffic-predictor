from datetime import datetime

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.storage.db import Base
from app.storage.models import Camera, TrafficMetric5m
from app.storage.repository import Repository


def test_upsert_metric_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        repo = Repository(session)
        repo.upsert_camera(
            Camera(
                camera_id="cam-1",
                source_url="http://example",
                intersection_id="int-1",
                direction="SN",
                is_enabled=True,
                roi_config={},
            )
        )
        ts = datetime(2026, 1, 1, 10, 0)
        repo.upsert_metric(
            TrafficMetric5m(
                timestamp_utc=ts,
                camera_id="cam-1",
                intersection_id="int-1",
                lane_id="all",
                vehicle_count=5,
                density=0.1,
                avg_speed_proxy=None,
                quality_score=0.9,
                source_image_uri="a.png",
            )
        )
        repo.upsert_metric(
            TrafficMetric5m(
                timestamp_utc=ts,
                camera_id="cam-1",
                intersection_id="int-1",
                lane_id="all",
                vehicle_count=9,
                density=0.2,
                avg_speed_proxy=None,
                quality_score=0.8,
                source_image_uri="b.png",
            )
        )
        session.commit()

        rows = repo.list_recent_metrics("cam-1")
        assert len(rows) == 1
        assert rows[0].vehicle_count == 9
        assert rows[0].source_image_uri == "b.png"
