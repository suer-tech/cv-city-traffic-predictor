from app.forecast.service import ForecastService


def test_congestion_probability() -> None:
    prob = ForecastService.congestion_probability(50, 100)
    assert prob == 0.5
