from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.model_config import TARGET, Experiment, get_experiments


DATA_PATH = Path("data/processed/canteen_visitors_processed.csv")
RESULTS_PATH = Path("report/experiments.csv")
MODEL_PATH = Path("models/best_model.joblib")


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        train = df[df["split"] == "train"].copy()
        val = df[df["split"] == "val"].copy()
        test = df[df["split"] == "test"].copy()
        return train, val, test

    dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))
    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train = df[pd.to_datetime(df["date"]).isin(train_dates)].copy()
    val = df[pd.to_datetime(df["date"]).isin(val_dates)].copy()
    test = df[pd.to_datetime(df["date"]).isin(test_dates)].copy()
    return train, val, test


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def score_model(model: Pipeline, data: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict[str, float]:
    x = data[num_cols + cat_cols]
    y = data[TARGET]
    pred = model.predict(x)
    mae = cast(float, mean_absolute_error(y, pred))
    mse = cast(float, mean_squared_error(y, pred))
    r2 = cast(float, r2_score(y, pred))

    return {
        "mae": mae,
        "rmse": mse**0.5,
        "r2": r2,
    }


def train_one(experiment: Experiment, train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, Pipeline]:
    name = experiment["name"]
    regressor = experiment["regressor"]
    num_cols = experiment["num_cols"]
    cat_cols = experiment["cat_cols"]

    model = Pipeline(
        [
            ("prep", make_preprocessor(num_cols, cat_cols)),
            ("model", regressor),
        ]
    )
    model.fit(train[num_cols + cat_cols], train[TARGET])

    scores = score_model(model, val, num_cols, cat_cols)
    row = {
        "model": name,
        "features": experiment["features"],
        "val_mae": round(scores["mae"], 3),
        "val_rmse": round(scores["rmse"], 3),
        "val_r2": round(scores["r2"], 4),
    }
    return row, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = cast(pd.DataFrame, pd.read_csv(args.data_path))
    train, val, test = split_by_date(df)

    rows = []
    trained_models = {}
    trained_features = {}
    for experiment in get_experiments():
        row, model = train_one(experiment, train, val)
        rows.append(row)
        name = experiment["name"]
        trained_models[name] = model
        trained_features[name] = (experiment["num_cols"], experiment["cat_cols"])

    results = pd.DataFrame(rows).sort_values("val_mae")
    best_name = results.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_num_cols, best_cat_cols = trained_features[best_name]
    test_scores = score_model(best_model, test, best_num_cols, best_cat_cols)

    results["test_mae"] = np.nan
    results["test_rmse"] = np.nan
    results["test_r2"] = np.nan
    best_idx = results["model"] == best_name
    results.loc[best_idx, "test_mae"] = round(test_scores["mae"], 3)
    results.loc[best_idx, "test_rmse"] = round(test_scores["rmse"], 3)
    results.loc[best_idx, "test_r2"] = round(test_scores["r2"], 4)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_path, index=False)
    joblib.dump(best_model, args.model_path)

    print(f"Train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}")
    print(results.to_string(index=False))
    print(f"Best model: {best_name}")
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
