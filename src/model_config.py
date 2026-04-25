from __future__ import annotations

from typing import TypedDict

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


SEED = 42
TARGET = "visitors_count"

BASE_NUM_COLS = [
    "avg_classes_today",
    "avg_classes_morning",
    "avg_classes_evening",
    "session_progress",
    "forecast_temperature",
    "forecast_precipitation_mm",
    "relative_humidity",
]

BASE_CAT_COLS = [
    "weekday",
    "month",
    "module_number",
    "building_id",
    "is_project_session",
]

NUM_COLS = BASE_NUM_COLS + [
    "classes_morning_share",
    "classes_evening_share",
]

CAT_COLS = BASE_CAT_COLS + [
    "is_weekend",
    "is_rainy",
    "is_strong_rain",
]


class Experiment(TypedDict):
    name: str
    regressor: object
    num_cols: list[str]
    cat_cols: list[str]
    features: str


def get_experiments() -> list[Experiment]:
    return [
        {
            "name": "dummy_mean",
            "regressor": DummyRegressor(strategy="mean"),
            "num_cols": BASE_NUM_COLS,
            "cat_cols": BASE_CAT_COLS,
            "features": "base",
        },
        {
            "name": "linear_regression_baseline",
            "regressor": LinearRegression(),
            "num_cols": BASE_NUM_COLS,
            "cat_cols": BASE_CAT_COLS,
            "features": "base",
        },
        {
            "name": "linear_regression_fe",
            "regressor": LinearRegression(),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
        {
            "name": "ridge_fe",
            "regressor": Ridge(alpha=1.0),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
        {
            "name": "knn_7_fe",
            "regressor": KNeighborsRegressor(n_neighbors=7),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
        {
            "name": "decision_tree_fe",
            "regressor": DecisionTreeRegressor(max_depth=8, random_state=SEED),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
        {
            "name": "gradient_boosting_fe",
            "regressor": GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.06,
                max_depth=3,
                random_state=SEED,
            ),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
        {
            "name": "random_forest_fe",
            "regressor": RandomForestRegressor(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1,
            ),
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "features": "feature_engineering",
        },
    ]
