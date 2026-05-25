from src.api import PredictRequest, health, predict


def test_health():
    assert health() == {"status": "ok"}


def test_predict_endpoint_function():
    request = PredictRequest(
        date="2026-04-25",
        building_id=1,
        avg_classes_today=4.0,
        avg_classes_morning=2.0,
        avg_classes_evening=1.0,
        forecast_temperature=12.5,
        forecast_precipitation_mm=1.2,
        relative_humidity=65.0,
    )

    response = predict(request)

    assert response.visitors_count > 0
