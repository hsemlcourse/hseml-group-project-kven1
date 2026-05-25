from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_sources import read_buildings
from src.inference import MODEL_PATH, PredictionInput, load_model, predict_visitors


BUILDINGS_PATH = Path("data/raw/buildings.csv")

app = FastAPI(title="Canteen Visitors Forecast API")


class PredictRequest(BaseModel):
    date: str = Field(..., examples=["2026-04-25"])
    building_id: int = Field(..., ge=1, examples=[1])
    avg_classes_today: float = Field(..., ge=0, examples=[4.0])
    avg_classes_morning: float = Field(..., ge=0, examples=[2.0])
    avg_classes_evening: float = Field(..., ge=0, examples=[1.0])
    forecast_temperature: float = Field(..., examples=[12.5])
    forecast_precipitation_mm: float = Field(..., ge=0, examples=[1.2])
    relative_humidity: float = Field(..., ge=0, le=100, examples=[65.0])


class PredictResponse(BaseModel):
    visitors_count: int


@lru_cache
def get_model() -> Any:
    return load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/buildings")
def buildings() -> list[dict[str, object]]:
    df = read_buildings(BUILDINGS_PATH)
    return df[["building_id", "address", "campus"]].to_dict(orient="records")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        data = PredictionInput(**request.model_dump())
        visitors = predict_visitors(get_model(), data)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(visitors_count=visitors)
