# Прогнозирование цен акций МосБиржи

## Описание проекта

Данный проект представляет собой ВКР, целью которой является прогнозирование цен акций на Московской бирже с использованием классических методов машинного обучения. Проект объединяет разнородные источники данных:
- **Технические индикаторы:** исторические данные цен, объём торгов, индикаторы (RSI, MACD, скользящие средние и т.д.);
- **Фундаментальный анализ:** финансовые отчёты, коэффициенты (P/E, P/B, ROE) и дивидендные показатели;
- **Макроэкономические данные:** курсы валют, цены на нефть, биржевые индексы, ключевая ставка;
- **Новостной и экспертный анализ:** обработка новостного потока и экспертных мнений с использованием LLM через API для оценки тональности.

Ключевыми задачами проекта являются как построение прогностической модели цен акций на основе всестороннего сбора, обработки и интеграции данных из различных источников, так и эффективная суммаризация этой разнородной информации (новостей, аналитики, экспертных мнений, финансовых отчетов) для её предоставления пользователям в сжатом, понятном и информативном виде.

## Особенности

- **Многоступенчатая ETL-система:** Сбор, очистка и синхронизация данных из разных источников.
- **Фичеринжиниринг:** Расчёт технических индикаторов, обработка фундаментальных данных и оценка новостных лент.
- **Моделирование:** Использование классических алгоритмов ML с учетом временных рядов.
- **Бэктестинг:** Историческая симуляция для оценки эффективности прогноза и анализа рисков.
- **Автоматизация:** Скрипты и оркестрация процессов (ETL, переобучение моделей) для регулярного обновления данных.

## Структура проекта

```
market-genius/
├── src/                       # Исходный код Python пакета
│   └── marketgenius/          # Корневой пакет (import marketgenius as mg)
│       ├── __init__.py
│       ├── core/              # Общие абстракции, utils, конфиги
│       │   ├── config.py      # Pydantic BaseSettings
│       │   ├── logging.py     # Инициализация логера
│       │   └── io.py          # helpers (s3, parquet, db)
│       ├── ingestion/         # Инкрементальная загрузка данных
│       │   ├── loaders/       # Каждый источник = модуль-loader
│       │   │   ├── moex.py
│       │   │   ├── cbr.py
│       │   │   └── telegram.py
│       │   └── __init__.py
│       ├── features/          # Фичеринжиниринг
│       │   ├── technical.py
│       │   ├── fundamental.py
│       │   └── sentiment.py
│       ├── models/            # ML/TS модели + backtesting
│       │   ├── baseline.py
│       │   ├── trainer.py
│       │   ├── predictor.py
│       │   └── evaluation.py
│       ├── pipelines/         # Prefect flows / Airflow DAGs (файлы-энтри)
│       │   ├── telegram_news.py
│       │   ├── macro_update.py
│       │   └── retrain_model.py
│       ├── services/          # FastAPI / gRPC сервисы
│       │   ├── api.py         # REST интерфейс /predict
│       │   ├── worker.py      # Celery/RQ worker
│       │   ├── telegram_publisher.py   # Публикует новости в TG-канал
│       │   └── telegram_query_bot.py  # Бот-интерфейс для запросов по фичам
│       └── cli.py             # Typer-CLI для dev-задач
│
├── infra/                     # Инфраструктурный код (IaC)
│   ├── docker/                # Мульти-Docker (api, worker, scheduler)
│   │   ├── api.Dockerfile
│   │   ├── worker.Dockerfile
│   │   └── base.Dockerfile
│   ├── k8s/                   # Helm-чарты или raw-манифесты
│   │   ├── api-deploy.yaml
│   │   ├── worker-deploy.yaml
│   │   ├── scheduler-deploy.yaml
│   │   └── ingress.yaml
│   └── terraform/             # Provisioning (S3, RDS, EKS…)
│
├── .github/
│   └── workflows/
│       ├── ci.yml             # linters + tests + build
│       └── cd.yml             # docker push + deploy
│
├── configs/                   # YAML/JSON конфиги (не код!)
│   ├── logging.yaml
│   ├── prefect.toml
│   ├── llm.yaml
│   └── channels.yaml
│
├── data/                      # 👀 Git-LFS или .gitignore 
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/                 # EDA и прототипирование
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   └── source/                # Sphinx-source, md-файлы
│
├── scripts/                   # Bash/PowerShell helpers (не путать с pipelines)
│   ├── run_etl.sh
│   ├── start_api.sh
│   └── deploy_k8s.sh
│
├── docker-compose.yml         # Локальный стек (api + worker + pg + grafana)
├── Makefile                   # make help – удобные алиасы
├── pyproject.toml             # poetry + tool configs
├── requirements.txt           # сгенерировано из poetry export (prod-deps)
├── .env.example               # шаблон секретов для dev
└── README.md
```

Более подробное описание структуры и её эволюции можно найти в [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) и [PLANNING.md](PLANNING.md).

## Архитектура проекта

Проект «Market Genius» разрабатывается с ориентацией на микросервисную архитектуру, удобство CI/CD и observability.

### Ключевые компоненты и слои:

1.  **Сбор данных (`src/marketgenius/ingestion/`)**: Модули для инкрементальной загрузки данных из различных источников (MOEX, CBR, Telegram и др.). Каждый источник представлен отдельным `loader`-ом.
2.  **Обработка и трансформация (ETL/Pipelines)**:
    *   **Пайплайны (`src/marketgenius/pipelines/`)**: Используются Prefect (или Airflow) для оркестрации потоков данных, таких как обработка новостей из Telegram (`telegram_news.py`), обновление макроэкономических показателей (`macro_update.py`) и переобучение моделей (`retrain_model.py`).
    *   **Фичеринжиниринг (`src/marketgenius/features/`)**: Генерация признаков для моделей, включая технические индикаторы, фундаментальные показатели и анализ тональности текста.
3.  **Машинное обучение (`src/marketgenius/models/`)**: Включает в себя разработку, обучение, оценку и бэктестинг ML/TS моделей.
4.  **Сервисы (`src/marketgenius/services/`)**:
    *   **API (`api.py`)**: REST-интерфейс на FastAPI для получения прогнозов (например, `/predict`).
    *   **Worker (`worker.py`)**: Фоновый обработчик задач (Celery/RQ) для выполнения ресурсоёмких операций.
    *   **Telegram Publisher (`telegram_publisher.py`)**: Бот для публикации саммари новостей и инсайтов в Telegram-канал.
    *   **Telegram Query Bot (`telegram_query_bot.py`)**: Интерактивный Telegram-бот для запроса информации из итоговых файлов с признаками.
5.  **Инфраструктура (`infra/`)**: Код для управления инфраструктурой (IaC).
    *   **Docker (`infra/docker/`)**: Dockerfiles для сборки образов (`api`, `worker`, `scheduler`, `telegram_publisher`, `telegram_query_bot`).
    *   **Kubernetes (`infra/k8s/`)**: Манифесты для развёртывания в Kubernetes.
    *   **Terraform (`infra/terraform/`)**: Опционально, для provisioning облачных ресурсов.
6.  **Данные (`data/`)**:
    *   `data/raw`: Неизменённые сырые данные.
    *   `data/processed`: Очищенные и предобработанные данные, включая `staging` и `features`.
7.  **Конфигурация (`configs/`, `core/config.py`)**: YAML/JSON файлы для настроек, которые читаются через Pydantic `BaseSettings`, с возможностью переопределения через переменные окружения.
8.  **CLI (`src/marketgenius/cli.py`)**: Интерфейс командной строки на Typer для вспомогательных dev-задач.

### Технологический стек (основные компоненты):

| Слой                | Инструменты/Библиотеки                     |
|---------------------|--------------------------------------------|
| ETL / Orchestration | Airflow                                    |
| Хранение данных     | Parquet + DuckDB / Postgres                |
| ML/TS модели        | scikit-learn, PyTorch, LightGBM            |
| NLP                 | OpenAI API, sentence-transformers          |
| API-слой            | FastAPI + Uvicorn                          |
| User Interaction    | Telegram Bot API (aiogram)                 |
| CI/CD               | GitHub Actions, Docker, Kubernetes (опц.)  |
| Контейнеризация     | Docker, Docker Compose                     |

### Ключевые принципы:

*   **Единый Python-пакет `marketgenius`**: Используется как зависимость во всех сервисах.
*   **Разделение кода и инфраструктуры**: `src/` для кода, `infra/` для IaC.
*   **Конфигурация через файлы и ENV**: Никаких hardcoded значений в коде.
*   **Observability**: `/health` и `/metrics` эндпоинты, структурированное JSON-логирование.
*   **CI/CD**: Автоматизация линтинга, тестов, сборки и деплоя.

## Контейнеры и их назначение:

| Образ              | Компонент                                  | Команда Entrypoint                                       |
|--------------------|--------------------------------------------|----------------------------------------------------------|
| `mg-api`           | FastAPI сервер                             | `uvicorn marketgenius.services.api:app --host 0.0.0.0 --port 8000` |
| `mg-worker`        | Celery/RQ задачи (ETL, retrain)            | `python -m marketgenius.services.worker`                   |
| `mg-scheduler`     | Prefect/Beat                               | `prefect agent start -q default`                         |
| `mg-telegram-pub`  | Telegram Publisher bot                     | `python -m marketgenius.services.telegram_publisher`       |
| `mg-telegram-bot`  | Telegram Query bot                         | `python -m marketgenius.services.telegram_query_bot`       |

## Дальнейшие шаги

План развития проекта, включая следующие спринты и цели, описан в [PLANNING.md](PLANNING.md).
