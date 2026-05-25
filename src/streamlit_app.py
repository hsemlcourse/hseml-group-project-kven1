from __future__ import annotations

import os
from datetime import date
from typing import Any, cast

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data
def load_buildings() -> list[dict[str, Any]]:
    response = requests.get(f"{API_URL}/buildings", timeout=5)
    response.raise_for_status()
    return cast(list[dict[str, Any]], response.json())


def main() -> None:
    st.set_page_config(page_title="Canteen visitors forecast")
    st.title("Прогноз посетителей столовой")

    try:
        buildings = load_buildings()
    except requests.RequestException as exc:
        st.error(f"API недоступен: {exc}")
        return

    building = st.selectbox(
        "Корпус",
        buildings,
        format_func=lambda item: f"{item['building_id']}. {item['address']}",
    )

    selected_date = st.date_input("Дата", value=date(2026, 4, 25))

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_classes_today = st.number_input("Среднее число пар", min_value=0.0, max_value=8.0, value=4.0, step=0.1)
    with col2:
        avg_classes_morning = st.number_input("Пары утром", min_value=0.0, max_value=8.0, value=2.0, step=0.1)
    with col3:
        avg_classes_evening = st.number_input("Пары вечером", min_value=0.0, max_value=8.0, value=1.0, step=0.1)

    col4, col5, col6 = st.columns(3)
    with col4:
        forecast_temperature = st.number_input("Температура, °C", value=12.0, step=0.5)
    with col5:
        forecast_precipitation_mm = st.number_input("Осадки, мм", min_value=0.0, value=0.0, step=0.1)
    with col6:
        relative_humidity = st.number_input("Влажность, %", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

    if st.button("Посчитать прогноз", type="primary"):
        payload = {
            "date": selected_date.isoformat(),
            "building_id": int(building["building_id"]),
            "avg_classes_today": avg_classes_today,
            "avg_classes_morning": avg_classes_morning,
            "avg_classes_evening": avg_classes_evening,
            "forecast_temperature": forecast_temperature,
            "forecast_precipitation_mm": forecast_precipitation_mm,
            "relative_humidity": relative_humidity,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Ошибка запроса: {exc}")
            return

        visitors = response.json()["visitors_count"]
        st.metric("Прогноз посетителей", visitors)


if __name__ == "__main__":
    main()
