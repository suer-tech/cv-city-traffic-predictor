from pathlib import Path

import pytest

pytest.importorskip("PIL")

from app.capture.service import CaptureService
from app.config import settings


def test_capture_synthetic(tmp_path: Path) -> None:
    old_root = settings.snapshot_root
    settings.snapshot_root = str(tmp_path)
    try:
        service = CaptureService()
        result = service.capture("cam-1", "http://example")
        assert result.provider in {"synthetic", "playwright"}
        assert Path(result.image_path).exists()
    finally:
        settings.snapshot_root = old_root
