from dataclasses import dataclass
from datetime import datetime

from app.config import settings
from app.storage.models import ExternalTrafficObservation


@dataclass
class ExternalTrafficResult:
    provider: str
    score: float | None
    status: str
    payload: dict


class EvaluationService:
    def pull_external_traffic(self, entity_type: str, entity_id: str) -> ExternalTrafficResult:
        if not settings.yandex_provider_enabled:
            return ExternalTrafficResult(
                provider=settings.yandex_provider_name,
                score=None,
                status="not_configured",
                payload={"reason": "provider_disabled"},
            )

        # Non-blocking stub provider until API key and legal extraction flow are configured.
        return ExternalTrafficResult(
            provider=settings.yandex_provider_name,
            score=settings.yandex_score_default,
            status="ok",
            payload={"source": "stub", "entity_type": entity_type, "entity_id": entity_id},
        )

    @staticmethod
    def as_model(ts: datetime, entity_type: str, entity_id: str, res: ExternalTrafficResult) -> ExternalTrafficObservation:
        return ExternalTrafficObservation(
            timestamp_utc=ts,
            entity_type=entity_type,
            entity_id=entity_id,
            provider=res.provider,
            traffic_score=res.score,
            source_status=res.status,
            raw_payload=res.payload,
        )
