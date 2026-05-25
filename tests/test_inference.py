from src.inference import PredictionInput, make_features
from src.model_config import CAT_COLS, NUM_COLS


def test_make_features_has_model_columns():
    data = PredictionInput(
        date="2026-04-25",
        building_id=1,
        avg_classes_today=4.0,
        avg_classes_morning=2.0,
        avg_classes_evening=1.0,
        forecast_temperature=12.5,
        forecast_precipitation_mm=1.2,
        relative_humidity=65.0,
    )

    features = make_features(data)

    assert list(features.columns) == NUM_COLS + CAT_COLS
    assert features.loc[0, "is_rainy"] == 1
    assert features.loc[0, "classes_morning_share"] == 0.5
