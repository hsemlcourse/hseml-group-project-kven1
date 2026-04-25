import pandas as pd

from src.preprocessing import prepare
from src.train_models import split_by_date


def test_prepare_adds_features():
    df = pd.DataFrame(
        {
            "visitors_count": [100, 120],
            "date": ["2025-01-10", "2025-01-11"],
            "weekday": [4, 5],
            "weekday_name": ["Friday", "Saturday"],
            "month": [1, 1],
            "module_number": [2, 2],
            "avg_classes_today": [4.0, 2.0],
            "avg_classes_morning": [2.0, 1.0],
            "avg_classes_evening": [1.0, 0.5],
            "session_progress": [0.5, 0.5],
            "forecast_temperature": [-2.0, -3.0],
            "forecast_precipitation_mm": [0.0, 9.0],
            "relative_humidity": [60.0, 80.0],
            "building_id": [1, 1],
            "building_address": ["34 Tallinskaya Ulitsa", "34 Tallinskaya Ulitsa"],
            "campus": ["Strogino / MIEM", "Strogino / MIEM"],
            "is_project_session": [0, 0],
        }
    )

    result = prepare(df)

    assert "is_weekend" in result.columns
    assert "classes_morning_share" in result.columns
    assert "split" in result.columns
    assert result.loc[1, "is_strong_rain"] == 1


def test_prepare_clips_target_only_on_train():
    dates = pd.date_range("2025-01-01", periods=20)
    visitors = [100] * 14 + [1000] * 6
    df = pd.DataFrame(
        {
            "visitors_count": visitors,
            "date": dates.astype(str),
            "weekday": dates.weekday,
            "weekday_name": dates.day_name(),
            "month": dates.month,
            "module_number": [2] * 20,
            "avg_classes_today": [4.0] * 20,
            "avg_classes_morning": [2.0] * 20,
            "avg_classes_evening": [1.0] * 20,
            "session_progress": [0.5] * 20,
            "forecast_temperature": [-2.0] * 20,
            "forecast_precipitation_mm": [0.0] * 20,
            "relative_humidity": [60.0] * 20,
            "building_id": [1] * 20,
            "building_address": ["34 Tallinskaya Ulitsa"] * 20,
            "campus": ["Strogino / MIEM"] * 20,
            "is_project_session": [0] * 20,
        }
    )

    result = prepare(df)
    holdout = result[result["split"].isin(["val", "test"])]

    assert holdout["visitors_count"].max() == 1000


def test_split_by_date_keeps_order():
    dates = pd.date_range("2025-01-01", periods=20)
    df = pd.DataFrame(
        {
            "date": dates.repeat(2).astype(str),
            "building_id": [1, 2] * 20,
            "visitors_count": range(40),
        }
    )

    train, val, test = split_by_date(df)

    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
