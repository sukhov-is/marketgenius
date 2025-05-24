# PLANNING.md

# 🔮 План развития проекта «Market Genius»

_(последнее обновление: 2025-05-20)_

## 1. Цели проекта

1.  Построить надёжную и воспроизводимую систему прогнозирования стоимости акций Московской биржи с горизонтом **1-180 дней**.
2.  Реализовать расширяемую архитектуру, допускающую добавление новых источников данных (новые API, CSV-фиды, базы данных).
3.  Обеспечить возможность регулярного, желательно **ежедневного**, автоматического обновления данных, переобучения моделей и публикации отчётов / метрик.
4.  Предоставить REST / CLI-интерфейс для получения прогнозов и мониторинга состояния системы.

## 2. Текущий статус кода

```
src/
├── data_ingestion/        # Загрузка сыровых временных рядов и макро-показателей
├── etl/                   # Многошаговый конвейер Telegram-новостей
├── utils/                 # Вспомогательные классы (GPT анализатор, чистка тикеров)
├── data_processing_temp/  # Временные standalone-скрипты (подлежат архивированию)
└── … 
```

+ Загрузка курсов валют, ключевой ставки, цен нефти уже реализована в `data_ingestion/`.
+ ETL-пайплайн новостей Telegram состоит из **7 нумерованных шагов** (`00_telegram_parser.py` … `06_process_batch_results.py`) и инкрементального `update_pipeline.py`.
+ Анализ новостей через OpenAI GPT вынесен в `utils/gpt_analyzer.py`, работает, но тесно связан с конфиг-JSON и может быть разбит на под-модули.

## 3. Целевая архитектура

### 3.1 Слой данных

1. **Raw Zone** (`data/raw`, `data/external`)  – неизменённые выгрузки из источников.
2. **Staging Zone** (`data/processed/staging`) – очищенные и синхронизированные данные, готовые к фичеринжинирингу.
3. **Feature Store** (`data/processed/features`) – итоговые фичи для моделей ML.
4. **Model Artifacts** (`artifacts/models`, `artifacts/metrics`) – веса моделей, pickle/onnx, отчёты
5. **Logs & Monitoring** (`logs/`, Prometheus + Grafana, optional)

### 3.2 Кодовая структура

```
src/
├── ingestion/           # <— НОВОЕ имя, вмещает data_ingestion/*
│   └── loaders/{moex, cbr, …}.py
├── pipelines/           # Пайплайны, orchestration (Airflow/Dagster Prefect)
│   ├── telegram_news.py
│   └── macro_update.py
├── features/            # Feature engineering (тех. индикаторы, NLP, …)
├── models/              # Обучение, инференс, backtesting
├── services/            # FastAPI сервисы, CLI, gRPC
├── core/                # Общие абстракции: configs, io, utils, logging
└── tests/
```

*Папки `feature_engineering/`, `models/` из текущего репо переедут в соответствующие каталоги.*

### 3.3 Технологический стек

| Слой                | Инструменты/Библиотеки |
|---------------------|-------------------------|
| ETL / Orchestration | Airflow |
| Хранение данных     | Parquet + DuckDB \| Postgres |
| ML/TS модели        | scikit-learn, pytorch, lightgbm |
| NLP                 | openai API |
| API-слой            | FastAPI + Uvicorn |
| User Interaction    | Telegram Bot API (aiogram) |
| CI/CD               | GitHub Actions, Docker, (Docker Compose / Kubernetes) |

## 4. Код-стайл и практики

1. **PEP-8 + Black** (line-length = 120).  В репозитории будет настроен `pre-commit`:
   * `black`, `isort`, `flake8`, `mypy`, `pytest`.
2. **Типизация**: обязательные `typing` аннотации + `mypy --strict` в CI.
3. **Документация**: docstring формата **Google** или **Numpy**; автогенерация `docs/` через Sphinx + Napoleon.
4. **Логирование**: стандартный `logging` с YAML-конфигом в `configs/logging.yaml`, формат json-line для продакшена.
5. **Тесты**: `pytest` + `pytest-cov`, целевой уровень покрытия ≥ 80 % для core-логики.
6. **Конфигурация**: Pydantic v2 `BaseSettings`; `.env` для секретов; все пути/параметры задаются через YAML/ENV, не «магическими» литералами.
7. **Версионирование данных/моделей**: DVC или MLflow;
8. **Секьюрити**: создание отдельных пользователей API-ключей;
9. **Чистота зависимостей**: `poetry` или `pip-tools` с `requirements.txt` + dependabot.

## 5. Ограничения и допущения

* **Данные Telegram** могут отсутствовать для исторических дат → backfill выполняется по «лучшей попытке»; модель должна уметь работать и без них.
* **Rate-Limit** API (ЦБ РФ, MOEX, OpenAI).  Доступ к OpenAI тарифицируется – кол-во запросов в день ограничено в `configs/llm.yaml`.
* **Запуск Windows**: скрипты должны работать кросс-платформенно, но heavy-ML предпочтительно запускать в Linux-контейнере.
* **Аппаратные ресурсы**: минимальная конфигурация – CPU 16 GB RAM; GPU опционально (pytorch fallback на CPU).
* **Лицензия**: MIT (если иное не указано).  Для финансовых данных соблюдаем лицензии поставщиков.

## 6. План рефакторинга (MVP → v1)

| Sprint | Цель | Выход | Ответственный |
|--------|------|-------|---------------|
| S-1 | Настройка CI/CD, pre-commit, Black + isort | зелёный pipeline | @dev |
| S-2 | Перемещение `data_ingestion/*` → `ingestion/`, унификация интерфейса `BaseLoader` | модуль `ingestion` | |
| S-3 | Разделить ETL-скрипты Telegram на классы + Prefect flow | `pipelines/telegram_news.py` | |
| S-4 | Базовый feature store и первый ML-базлайн (LightGBM) | ноутбук + `models/baseline.py` | |
| S-5 | Создать FastAPI endpoint `/predict` + docker-compose | образ `market-genius-api` | |
| S-6 | Реализовать Telegram Publisher + Query Bot | контейнеры `mg-telegram-pub`, `mg-telegram-bot` | |

## 7. Метрики успеха

1. **MAPE ≤ 5 %** на тестовом периоде 2023-2024.
2. Среднее время обработки дневного инкремента ≤ 30 мин.
3. Покрытие unit-тестами 80 % + 0 Critical/Security Issues (Bandit).
4. End-to-end latency REST запроса `/predict` ≤ 200 мс.

---

_Документ является живым._