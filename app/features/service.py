from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeatureVector:
    hour: int
    day_of_week: int
    is_weekend: int
    lag_1: float
    lag_2: float
    lag_3: float
    roll_3: float


def floor_to_5m(ts: datetime) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)


def build_feature_vector(values: list[float], ts: datetime) -> FeatureVector:
    if not values:
        values = [0.0]
    lag_1 = values[-1]
    lag_2 = values[-2] if len(values) > 1 else lag_1
    lag_3 = values[-3] if len(values) > 2 else lag_2
    tail = values[-3:]
    roll_3 = sum(tail) / len(tail)

    return FeatureVector(
        hour=ts.hour,
        day_of_week=ts.weekday(),
        is_weekend=int(ts.weekday() >= 5),
        lag_1=lag_1,
        lag_2=lag_2,
        lag_3=lag_3,
        roll_3=roll_3,
    )
