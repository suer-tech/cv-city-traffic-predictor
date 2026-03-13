from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

from PIL import Image, ImageDraw

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    camera_id: str
    capture_ts_utc: datetime
    image_path: str
    persisted_hourly: bool
    provider: str


class CaptureService:
    """Capture service with provider fallback.

    Providers:
    - playwright (if installed)
    - synthetic (always available)
    """

    def capture(self, camera_id: str, source_url: str) -> CaptureResult:
        now = datetime.utcnow().replace(second=0, microsecond=0)
        ephemeral_dir = Path(settings.snapshot_root) / "ephemeral" / camera_id
        hourly_dir = Path(settings.snapshot_root) / "hourly" / camera_id
        ephemeral_dir.mkdir(parents=True, exist_ok=True)
        hourly_dir.mkdir(parents=True, exist_ok=True)

        image_name = now.strftime("%Y%m%d_%H%M.png")
        ep_path = ephemeral_dir / image_name

        provider = self._capture_provider(camera_id=camera_id, source_url=source_url, target_path=ep_path, ts=now)

        persisted_hourly = now.minute % settings.persist_snapshot_every_minutes == 0
        if persisted_hourly:
            hourly_path = hourly_dir / image_name
            Image.open(ep_path).save(hourly_path)
            image_uri = str(hourly_path)
            try:
                ep_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("failed_to_cleanup_ephemeral", extra={"path": str(ep_path)})
        else:
            image_uri = str(ep_path)

        logger.info(
            "capture_completed",
            extra={"camera_id": camera_id, "image": image_uri, "provider": provider, "hourly": persisted_hourly},
        )
        return CaptureResult(
            camera_id=camera_id,
            capture_ts_utc=now,
            image_path=image_uri,
            persisted_hourly=persisted_hourly,
            provider=provider,
        )

    def capture_with_retries(self, camera_id: str, source_url: str) -> CaptureResult | None:
        for attempt in range(1, settings.capture_retry_count + 1):
            try:
                return self.capture(camera_id, source_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "capture_failed_attempt",
                    extra={"camera_id": camera_id, "attempt": attempt, "error": str(exc)},
                )
        logger.error("capture_failed_exhausted", extra={"camera_id": camera_id})
        return None

    def _capture_provider(self, camera_id: str, source_url: str, target_path: Path, ts: datetime) -> str:
        provider = settings.capture_provider.lower()
        if provider == "playwright":
            ok = self._capture_with_playwright(source_url, target_path)
            if ok:
                return "playwright"
            logger.warning("playwright_capture_unavailable_fallback", extra={"camera_id": camera_id})

        self._generate_placeholder_frame(target_path, camera_id, ts, source_url)
        return "synthetic"

    @staticmethod
    def _capture_with_playwright(source_url: str, target_path: Path) -> bool:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError:
            return False

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(source_url, wait_until="networkidle", timeout=settings.capture_timeout_ms)
                page.screenshot(path=str(target_path), full_page=False)
                browser.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _generate_placeholder_frame(path: Path, camera_id: str, ts: datetime, source_url: str) -> None:
        img = Image.new("RGB", (1280, 720), color=(35, 35, 35))
        draw = ImageDraw.Draw(img)
        draw.rectangle((100, 150, 1180, 650), outline=(20, 200, 20), width=4)
        draw.text((120, 160), f"camera={camera_id}", fill=(255, 255, 255))
        draw.text((120, 185), f"ts={ts.isoformat()}Z", fill=(255, 255, 255))
        draw.text((120, 210), source_url, fill=(180, 180, 180))

        base_x = 180 + (ts.minute % 10) * 15
        for idx in range(6):
            x1 = base_x + idx * 150
            draw.rectangle((x1, 520, x1 + 55, 555), fill=(220, 220, 0))

        img.save(path)
