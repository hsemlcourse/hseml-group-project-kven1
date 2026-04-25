from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/processed/canteen_visitors_processed.csv")
OUT_DIR = Path("report/images")


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def visitors_hist(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["visitors_count"], bins=35, kde=True)
    plt.title("Distribution of visitors count")
    plt.xlabel("Visitors count")
    save_plot(out_dir / "visitors_distribution.png")


def visitors_by_building(df: pd.DataFrame, out_dir: Path) -> None:
    data = df.groupby("building_address", as_index=False)["visitors_count"].mean()
    data = data.sort_values("visitors_count", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="visitors_count", y="building_address", hue="building_address", legend=False)
    plt.title("Average visitors by building")
    plt.xlabel("Average visitors")
    plt.ylabel("")
    save_plot(out_dir / "visitors_by_building.png")


def visitors_by_weekday(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="weekday_name", y="visitors_count", order=order, errorbar=None)
    plt.title("Average visitors by weekday")
    plt.xlabel("")
    plt.ylabel("Average visitors")
    plt.xticks(rotation=35)
    save_plot(out_dir / "visitors_by_weekday.png")


def visitors_vs_classes(df: pd.DataFrame, out_dir: Path) -> None:
    sample = df.sample(min(len(df), 2000), random_state=42)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="avg_classes_today", y="visitors_count", hue="campus", alpha=0.45)
    plt.title("Visitors and average classes today")
    plt.xlabel("Average classes today")
    plt.ylabel("Visitors count")
    plt.legend(fontsize=7)
    save_plot(out_dir / "visitors_vs_classes.png")


def visitors_vs_temperature(df: pd.DataFrame, out_dir: Path) -> None:
    sample = df.sample(min(len(df), 2000), random_state=42)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="forecast_temperature", y="visitors_count", alpha=0.35)
    sns.regplot(data=sample, x="forecast_temperature", y="visitors_count", scatter=False, color="red")
    plt.title("Visitors and temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Visitors count")
    save_plot(out_dir / "visitors_vs_temperature.png")


def corr_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "visitors_count",
        "avg_classes_today",
        "avg_classes_morning",
        "avg_classes_evening",
        "session_progress",
        "forecast_temperature",
        "forecast_precipitation_mm",
        "relative_humidity",
        "classes_morning_share",
        "classes_evening_share",
    ]

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation matrix")
    save_plot(out_dir / "correlation_matrix.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = cast(pd.DataFrame, pd.read_csv(args.data_path))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    visitors_hist(df, args.out_dir)
    visitors_by_building(df, args.out_dir)
    visitors_by_weekday(df, args.out_dir)
    visitors_vs_classes(df, args.out_dir)
    visitors_vs_temperature(df, args.out_dir)
    corr_heatmap(df, args.out_dir)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
