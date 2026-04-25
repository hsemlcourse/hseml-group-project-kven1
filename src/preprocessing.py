from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/canteen_visitors_synthetic.csv")
OUT_PATH = Path("data/processed/canteen_visitors_processed.csv")
TARGET = "visitors_count"
CATEGORICAL_COLS = ["weekday_name", "building_address", "campus"]
INT_COLS = ["weekday", "month", "module_number", "building_id", "is_project_session"]
SPLIT_COL = "split"


def add_time_split(df: pd.DataFrame) -> pd.DataFrame:
    dates = np.array(sorted(df["date"].unique()))
    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])

    df = df.copy()
    df[SPLIT_COL] = "test"
    df.loc[df["date"].isin(train_dates), SPLIT_COL] = "train"
    df.loc[df["date"].isin(val_dates), SPLIT_COL] = "val"
    return df


def train_target_clip_bounds(train: pd.DataFrame) -> dict[int, tuple[float, float]]:
    bounds = {}
    for building_id, part in train.groupby("building_id"):
        q1 = part[TARGET].quantile(0.25)
        q3 = part[TARGET].quantile(0.75)
        iqr = q3 - q1
        low = max(0, q1 - 1.5 * iqr)
        high = q3 + 1.5 * iqr
        bounds[int(building_id)] = (low, high)
    return bounds


def clip_train_target(df: pd.DataFrame, bounds: dict[int, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    train_mask = df[SPLIT_COL] == "train"

    for building_id, (low, high) in bounds.items():
        mask = train_mask & (df["building_id"] == building_id)
        df.loc[mask, TARGET] = df.loc[mask, TARGET].clip(low, high)

    df[TARGET] = df[TARGET].round().astype(int)
    return df


def train_fill_values(train: pd.DataFrame, num_cols: list[str]) -> tuple[dict[str, float], dict[str, object]]:
    numeric_fill = {col: float(train[col].median()) for col in num_cols}
    categorical_fill = {}

    for col in CATEGORICAL_COLS:
        mode = train[col].mode(dropna=True)
        categorical_fill[col] = mode.iloc[0] if not mode.empty else "unknown"

    return numeric_fill, categorical_fill


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    df["date"] = pd.to_datetime(df["date"])
    for col in INT_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = add_time_split(df)

    train = df[df[SPLIT_COL] == "train"]
    num_cols = [col for col in df.select_dtypes(include="number").columns if col != TARGET]
    numeric_fill, categorical_fill = train_fill_values(train, num_cols)

    for col in num_cols:
        df[col] = df[col].fillna(numeric_fill[col])
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(categorical_fill[col])

    for col in INT_COLS:
        df[col] = df[col].round().astype(int)

    df = clip_train_target(df, train_target_clip_bounds(df[df[SPLIT_COL] == "train"]))

    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rainy"] = (df["forecast_precipitation_mm"] > 0).astype(int)
    df["is_strong_rain"] = (df["forecast_precipitation_mm"] >= 8).astype(int)
    df["classes_morning_share"] = np.where(
        df["avg_classes_today"] > 0,
        df["avg_classes_morning"] / df["avg_classes_today"],
        0,
    )
    df["classes_evening_share"] = np.where(
        df["avg_classes_today"] > 0,
        df["avg_classes_evening"] / df["avg_classes_today"],
        0,
    )

    df["date"] = df["date"].dt.date.astype(str)
    return df.sort_values(["date", "building_id"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = cast(pd.DataFrame, pd.read_csv(args.input_path))
    df = prepare(df)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    print(f"Saved {len(df)} rows to {args.output_path}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
