import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import settings
from app.orchestration.jobs import capture_and_extract_job, inference_job, retrain_job

logger = logging.getLogger(__name__)


_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        capture_and_extract_job,
        "interval",
        minutes=settings.capture_interval_minutes,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        inference_job,
        "interval",
        minutes=settings.inference_interval_minutes,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        retrain_job,
        "interval",
        hours=settings.retrain_interval_hours,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info("scheduler_started")
    return scheduler


def run_forever() -> None:
    scheduler = start_scheduler()
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        scheduler.shutdown()
