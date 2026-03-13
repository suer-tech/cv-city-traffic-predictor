from datetime import datetime

from app.features.service import build_feature_vector, floor_to_5m


def test_floor_to_5m() -> None:
    ts = datetime(2026, 1, 1, 12, 13, 40)
    assert floor_to_5m(ts) == datetime(2026, 1, 1, 12, 10)


def test_build_feature_vector() -> None:
    fv = build_feature_vector([2, 4, 8, 10], datetime(2026, 1, 1, 12, 0, 0))
    assert fv.lag_1 == 10
    assert fv.lag_2 == 8
    assert fv.roll_3 == (4 + 8 + 10) / 3
