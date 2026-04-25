from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


SEED = 42
DAY_EFFECT = {0: -20, 1: 0, 2: 10, 3: 10, 4: -15, 5: -80, 6: -120}
DAY_MULT = {0: 0.92, 1: 1.03, 2: 1.08, 3: 1.08, 4: 0.86, 5: 0.35, 6: 0.12}

OUTPUT_COLUMNS = [
    "visitors_count",
    "date",
    "weekday",
    "weekday_name",
    "month",
    "module_number",
    "avg_classes_today",
    "avg_classes_morning",
    "avg_classes_evening",
    "session_progress",
    "forecast_temperature",
    "forecast_precipitation_mm",
    "relative_humidity",
    "building_id",
    "building_address",
    "campus",
    "is_project_session",
]


def module_num(date: pd.Timestamp) -> int:
    if date.month in (9, 10) or (date.month == 11 and date.day <= 5):
        return 1
    if date.month in (11, 12) or (date.month == 1 and date.day <= 20):
        return 2
    if date.month in (1, 2, 3):
        return 3
    return 4


def session_progress(date: pd.Timestamp) -> float:
    year = date.year
    periods = [
        (pd.Timestamp(year - 1, 9, 1), pd.Timestamp(year - 1, 11, 5)),
        (pd.Timestamp(year - 1, 11, 6), pd.Timestamp(year, 1, 20)),
        (pd.Timestamp(year, 1, 21), pd.Timestamp(year, 3, 31)),
        (pd.Timestamp(year, 4, 1), pd.Timestamp(year, 6, 20)),
        (pd.Timestamp(year, 9, 1), pd.Timestamp(year, 11, 5)),
        (pd.Timestamp(year, 11, 6), pd.Timestamp(year + 1, 1, 20)),
    ]
    for start, end in periods:
        if start <= date <= end:
            return round((date - start).days / (end - start).days, 3)
    return 0.0


def is_project_session(date: pd.Timestamp) -> int:
    year = date.year
    periods = [
        (pd.Timestamp(year, 1, 15), pd.Timestamp(year, 1, 28)),
        (pd.Timestamp(year, 3, 25), pd.Timestamp(year, 4, 7)),
        (pd.Timestamp(year, 6, 10), pd.Timestamp(year, 6, 25)),
        (pd.Timestamp(year, 11, 1), pd.Timestamp(year, 11, 14)),
    ]
    return int(any(start <= date <= end for start, end in periods))


def activity(date: pd.Timestamp) -> float:
    if date.weekday() == 6:
        return 0.08
    if date.weekday() == 5:
        return 0.18
    if date.month in (7, 8):
        return 0.12
    if date.month == 1 and date.day <= 8:
        return 0.10
    if date.month == 5 and date.day <= 9:
        return 0.45
    return 1.0


def generate_classes(date: pd.Timestamp, building: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    mean = building["max_classes"] * activity(date) * DAY_MULT[date.weekday()]
    if is_project_session(date):
        mean *= 0.72

    total = float(np.clip(rng.normal(mean, 0.45), 0, building["max_classes"] + 1))
    morning = total * float(np.clip(rng.normal(0.48, 0.09), 0.25, 0.70))
    evening = total * float(np.clip(rng.normal(0.22, 0.07), 0.05, 0.45))
    return round(total, 2), round(morning, 2), round(evening, 2)


def temperature_effect(temp: float) -> int:
    if temp < -12 or temp > 28:
        return -14
    if temp < -3:
        return -6
    if temp > 22:
        return -4
    return 0


def visitors_count(row: dict, building: pd.Series, rng: np.random.Generator) -> int:
    value = (
        building["base_flow"]
        + DAY_EFFECT[row["weekday"]]
        + building["class_weight"] * row["avg_classes_today"]
        + building["morning_weight"] * row["avg_classes_morning"]
        + building["evening_weight"] * row["avg_classes_evening"]
        + building["session_weight"] * row["session_progress"]
        + building["project_session_effect"] * row["is_project_session"]
        + temperature_effect(row["forecast_temperature"])
        + (building["precipitation_effect"] if row["forecast_precipitation_mm"] >= 8 else 0)
        + rng.normal(0, building["noise_std"])
    )

    if row["weekday"] >= 5:
        value *= building["weekend_factor"]
    if row["month"] in (7, 8):
        value *= 0.45

    return int(max(round(value), 5))


def make_daily_building_row(
    date: pd.Timestamp,
    weather_row: pd.Series,
    building: pd.Series,
    rng: np.random.Generator,
) -> dict:
    total, morning, evening = generate_classes(date, building, rng)
    row = {
        "date": date.date().isoformat(),
        "weekday": date.weekday(),
        "weekday_name": date.day_name(),
        "month": date.month,
        "module_number": module_num(date),
        "avg_classes_today": total,
        "avg_classes_morning": morning,
        "avg_classes_evening": evening,
        "session_progress": session_progress(date),
        "forecast_temperature": round(float(weather_row["forecast_temperature"]), 1),
        "forecast_precipitation_mm": round(float(weather_row["forecast_precipitation_mm"]), 1),
        "relative_humidity": round(float(weather_row["relative_humidity"]), 1),
        "building_id": int(building["building_id"]),
        "building_address": building["address"],
        "campus": building["campus"],
        "is_project_session": is_project_session(date),
    }
    row["visitors_count"] = visitors_count(row, building, rng)
    return row


def make_dataset(weather: pd.DataFrame, buildings: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _, weather_row in weather.iterrows():
        date = cast(pd.Timestamp, pd.Timestamp(weather_row["date"]))
        for _, building in buildings.iterrows():
            rows.append(make_daily_building_row(date, weather_row, building, rng))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
