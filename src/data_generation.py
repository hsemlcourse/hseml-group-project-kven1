from __future__ import annotations

import argparse
from pathlib import Path

from src.data_sources import read_buildings, read_weather
from src.synthetic_canteen import SEED, make_dataset


WEATHER_PATH = Path("data/raw/open-meteo-55.75N37.62E149m.csv")
BUILDINGS_PATH = Path("data/raw/buildings.csv")
OUT_PATH = Path("data/raw/canteen_visitors_synthetic.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather-path", type=Path, default=WEATHER_PATH)
    parser.add_argument("--buildings-path", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather = read_weather(args.weather_path)
    buildings = read_buildings(args.buildings_path)
    df = make_dataset(weather, buildings, seed=args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    print(f"Saved {len(df)} rows to {args.output_path}")
    print(f"Dates: {df['date'].min()} - {df['date'].max()}")


if __name__ == "__main__":
    main()
