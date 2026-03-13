from pathlib import Path

import pytest

pytest.importorskip("PIL")

from app.capture.service import CaptureService
from app.config import settings


def test_capture_synthetic(tmp_path: Path) -> None:
    old_root = settings.snapshot_root
    old_provider = settings.capture_provider
    old_fallback = settings.capture_allow_synthetic_fallback
    settings.snapshot_root = str(tmp_path)
    settings.capture_provider = "synthetic"
    settings.capture_allow_synthetic_fallback = True
    try:
        service = CaptureService()
        result = service.capture("cam-1", "http://example")
        assert result.provider == "synthetic"
        assert Path(result.image_path).exists()
    finally:
        settings.snapshot_root = old_root
        settings.capture_provider = old_provider
        settings.capture_allow_synthetic_fallback = old_fallback
