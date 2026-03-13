from dataclasses import asdict
from datetime import datetime
import logging

from app.features.service import build_feature_vector

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    mlflow = None


class ForecastService:
    """CPU-light forecasting service with baseline rolling logic.

    Keeps interface compatible for future model upgrade.
    """

    def __init__(self) -> None:
        self.horizon_bias: dict[int, float] = {10: 0.0, 20: 0.0, 30: 0.0}

    def train(self, history: list[tuple[datetime, float]]) -> None:
        if len(history) < 16:
            return

        values = [v for _, v in history]
        run_ctx = (
            mlflow.start_run(run_name="daily_retrain", nested=True)
            if mlflow is not None
            else _NoopContextManager()
        )
        with run_ctx:
            for horizon in (10, 20, 30):
                steps = horizon // 5
                deltas: list[float] = []
                for i in range(3, len(values) - steps):
                    base = sum(values[max(0, i - 2) : i + 1]) / min(i + 1, 3)
                    target = values[i + steps]
                    deltas.append(target - base)
                self.horizon_bias[horizon] = sum(deltas) / len(deltas) if deltas else 0.0
                if mlflow is not None:
                    mlflow.log_metric(f"train_samples_h{horizon}", len(deltas))
                else:
                    logger.info("train_samples", extra={"horizon": horizon, "samples": len(deltas)})

    def predict(self, history: list[tuple[datetime, float]], now: datetime) -> dict[int, float]:
        values = [v for _, v in history] or [0.0]
        fv = build_feature_vector(values, now)
        baseline = fv.roll_3

        preds: dict[int, float] = {}
        for horizon in (10, 20, 30):
            pred = baseline + self.horizon_bias.get(horizon, 0.0)
            preds[horizon] = max(float(pred), 0.0)
        return preds

    @staticmethod
    def congestion_probability(pred_count: float, threshold: int) -> float:
        if threshold <= 0:
            return 0.0
        return float(min(max(pred_count / threshold, 0.0), 1.0))


class _NoopContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
