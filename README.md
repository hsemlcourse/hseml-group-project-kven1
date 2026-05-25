[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Прогноз количества посетителей столовой

**Студент:** Удалов Семён Борисович

**Группа:** 237


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Деплой](#деплой)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** регрессия.

**Датасет:** синтетический датасет посещаемости столовых ВШЭ. Для погодных признаков используется CSV из Open-Meteo, для корпусов — отдельный справочник `data/raw/buildings.csv`.

**Целевая метрика:** MAE. Она выбрана как основная, потому что показывает среднюю ошибку прогноза в количестве посетителей. Дополнительно считаются RMSE и R2.


## Структура репозитория
```
.
├── data
│   ├── raw                     # Open-Meteo CSV, справочник корпусов, синтетический датасет
│   └── processed               # Обработанный датасет
├── models                      # Сохранённая лучшая модель
├── notebooks                   # Зарезервировано под ноутбуки, на CP1 не используется
├── presentation                # Материалы для презентации
├── report
│   ├── images                  # Графики EDA
│   ├── data_generation.md      # Описание генерации данных
│   ├── experiments.csv         # Таблица экспериментов
│   └── report.md               # Отчёт
├── src
│   ├── data_generation.py      # CLI генерации синтетических данных
│   ├── data_sources.py         # Чтение Open-Meteo CSV и справочника корпусов
│   ├── synthetic_canteen.py    # Формулы генерации признаков и target
│   ├── preprocessing.py        # Предобработка данных
│   ├── make_plots.py           # EDA-графики
│   ├── model_config.py         # Признаки и конфиги экспериментов
│   ├── inference.py            # Подготовка признаков для прогноза
│   ├── api.py                  # FastAPI-сервис
│   ├── streamlit_app.py        # Web-интерфейс
│   └── train_models.py         # Обучение и оценка моделей
├── tests
│   ├── test_pipeline.py        # Тесты пайплайна
│   ├── test_inference.py       # Тест подготовки признаков для inference
│   └── test_api.py             # Тест API-обёртки
├── .github/workflows/ci.yml    # CI с проверками ruff и flake8
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

CP1 pipeline:

```bash
python -m src.data_generation
python -m src.preprocessing
python -m src.make_plots
python -m src.train_models
```

После запуска создаются:

```text
data/raw/canteen_visitors_synthetic.csv
data/processed/canteen_visitors_processed.csv
report/images/*.png
report/experiments.csv
models/best_model.joblib
```

## Данные
- `data/raw/open-meteo-55.75N37.62E149m.csv` — погодные данные Open-Meteo Historical Forecast API для Москвы
- `data/raw/buildings.csv` — список корпусов и параметры генерации
- `data/raw/canteen_visitors_synthetic.csv` — синтетический датасет, создаётся скриптом
- `data/processed/canteen_visitors_processed.csv` — предобработанные данные, создаётся скриптом

Погодный CSV скачан по запросу:

```text
https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=55.7558&longitude=37.6176&start_date=2024-04-25&end_date=2026-04-25&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum&timezone=Europe%2FMoscow&format=csv
```

Из Open-Meteo используются дневная средняя температура, дневная средняя относительная влажность и сумма осадков за день.

Одна строка датасета соответствует одному корпусу в один день.

На этапе preprocessing выполняются:
- удаление дублей
- приведение типов
- заполнение пропусков
- обработка выбросов по `visitors_count`
- создание признаков `is_weekend`, `is_rainy`, `is_strong_rain`, `classes_morning_share`, `classes_evening_share`


## Результаты
Основная метрика - MAE на validation. Полная таблица после запуска сохраняется в `report/experiments.csv`.

| Модель | Признаки | MAE val |
|--------|----------|--------:|
| DummyRegressor | base | 118.248 |
| LinearRegression baseline | base | 23.842 |
| LinearRegression FE | base + FE | 23.586 |
| Ridge FE | base + FE | 23.610 |
| KNNRegressor FE | base + FE | 26.410 |
| DecisionTreeRegressor FE | base + FE | 14.560 |
| GradientBoostingRegressor FE | base + FE | 13.216 |
| RandomForestRegressor FE | base + FE | 11.426 |

Лучшая модель на validation - `RandomForestRegressor FE`. На test: MAE `11.827`, RMSE `16.878`, R2 `0.9849`.

Графики лежат в `report/images/`.


## Деплой

Перед запуском деплоя должна быть обученная модель `models/best_model.joblib`. Если её нет, сначала запустите пайплайн обучения.

```bash
docker compose up --build
```

После запуска:
- API: `http://localhost:8000`
- Swagger-документация API: `http://localhost:8000/docs`
- Web-интерфейс: `http://localhost:8501`

Пример запроса к API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-04-25",
    "building_id": 1,
    "avg_classes_today": 4.0,
    "avg_classes_morning": 2.0,
    "avg_classes_evening": 1.0,
    "forecast_temperature": 12.5,
    "forecast_precipitation_mm": 1.2,
    "relative_humidity": 65.0
  }'
```


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)

Описание генерации данных: [`report/data_generation.md`](report/data_generation.md)
