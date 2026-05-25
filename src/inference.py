from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from src.model_config import CAT_COLS, NUM_COLS
from src.synthetic_canteen import is_project_session, module_num, session_progress


MODEL_PATH = Path("models/best_model.joblib")


@dataclass(frozen=True)
class PredictionInput:
    date: str
    building_id: int
    avg_classes_today: float
    avg_classes_morning: float
    avg_classes_evening: float
    forecast_temperature: float
    forecast_precipitation_mm: float
    relative_humidity: float


def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def make_features(data: PredictionInput) -> pd.DataFrame:
    date = cast(pd.Timestamp, pd.Timestamp(data.date))
    if pd.isna(date):
        raise ValueError("Bad date format")

    morning_share = data.avg_classes_morning / data.avg_classes_today if data.avg_classes_today > 0 else 0.0
    evening_share = data.avg_classes_evening / data.avg_classes_today if data.avg_classes_today > 0 else 0.0

    row = {
        "avg_classes_today": data.avg_classes_today,
        "avg_classes_morning": data.avg_classes_morning,
        "avg_classes_evening": data.avg_classes_evening,
        "session_progress": session_progress(date),
        "forecast_temperature": data.forecast_temperature,
        "forecast_precipitation_mm": data.forecast_precipitation_mm,
        "relative_humidity": data.relative_humidity,
        "classes_morning_share": morning_share,
        "classes_evening_share": evening_share,
        "weekday": date.weekday(),
        "month": date.month,
        "module_number": module_num(date),
        "building_id": data.building_id,
        "is_project_session": is_project_session(date),
        "is_weekend": int(date.weekday() >= 5),
        "is_rainy": int(data.forecast_precipitation_mm > 0),
        "is_strong_rain": int(data.forecast_precipitation_mm >= 8),
    }
    return pd.DataFrame([row], columns=NUM_COLS + CAT_COLS)


def predict_visitors(model: Any, data: PredictionInput) -> int:
    features = make_features(data)
    prediction = float(model.predict(features)[0])
    return int(max(round(prediction), 0))
