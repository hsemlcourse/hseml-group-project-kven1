from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


WEATHER_COLUMNS = [
    "date",
    "forecast_temperature",
    "forecast_precipitation_mm",
    "relative_humidity",
]

BUILDING_COLUMNS = [
    "building_id",
    "address",
    "campus",
    "base_flow",
    "class_weight",
    "morning_weight",
    "evening_weight",
    "session_weight",
    "project_session_effect",
    "precipitation_effect",
    "noise_std",
    "max_classes",
    "weekend_factor",
]


def read_weather(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path, skiprows=3))
    df = df.rename(
        columns={
            "time": "date",
            "temperature_2m_mean (°C)": "forecast_temperature",
            "relative_humidity_2m_mean (%)": "relative_humidity",
            "precipitation_sum (mm)": "forecast_precipitation_mm",
        }
    )
    if any(col not in df.columns for col in WEATHER_COLUMNS):
        raise ValueError("Bad weather CSV format")

    df = df[WEATHER_COLUMNS]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def read_buildings(path: Path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path))
    if any(col not in df.columns for col in BUILDING_COLUMNS):
        raise ValueError("Bad buildings CSV format")
    return df[BUILDING_COLUMNS]
