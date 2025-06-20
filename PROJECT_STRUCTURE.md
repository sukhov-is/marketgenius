# PROJECT_STRUCTURE.md

# 🗂️ Структура репозитория «Market Genius»

Ниже приведена opinionated-структура, ориентированная на:
• микросервисное развёртывание (Docker/Kubernetes),  
• observability (Prometheus, Grafana, логирование),  
• удобство CI/CD (GitHub Actions) и локальной разработки.

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

## Ключевые идеи организации

1. **Единый Python-пакет** `marketgenius`, импортируемый как зависимость во всех сервисах.  Установка: `pip install -e .` (editable).
2. **infra/** содержит всё, что связано с окружением (docker, k8s, terraform).  _Код отдельно – инфраструктура отдельно._
3. **Конфиги** хранятся в `configs/`, код их не импортирует напрямую, а читает через `core.config.Config` (Pydantic + `env_override`).
4. **Data outside Git**: большой объём – через DVC или S3; `data/` игнорируется либо подключён Git-LFS.
5. **Makefile** даёт cross-platform алиасы (`make dev`, `make test`, `make build-api`).
6. **Docker multi-stage**: базовый образ `python:3.12-slim` + слой poetry deps + copy only нужные файлы (плюс `--platform=linux/amd64`).
7. **Observability**: каждый сервис (api, worker, scheduler) имеет:
   - `/health` HTTP probe,
   - `/metrics` с Prometheus exporter,
   - структурированные JSON-логи.
8. **CI/CD**: `ci.yml` → lint+pytest, `cd.yml` → build & push образы, `kubectl rollout`.
9. **Модульные тесты** хранятся рядом с кодом (`tests/unit`) + фикстуры в `tests/conftest.py`.
10. **IaC=код** → изменения в `infra/*` проходят через Pull Request и terraform-plan в CI.

## Файлы верхнего уровня

| Файл | Назначение |
|------|------------|
| `docker-compose.yml` | Поднимает стек для локальной разработки (Postgres, MinIO, Prometheus, API, worker). |
| `pyproject.toml` | Управляет зависимостями и конфигурациями линтеров. |
| `Makefile` | Сборник команд (`make lint`, `make fmt`, `make run-api`). |
| `.env.example` | Перечень требуемых секретов (API ключи, DSN). |

## Контейнеры

| Образ | Компонент | Команда Entrypoint |
|-------|-----------|--------------------|
| `mg-api` | FastAPI сервер | `uvicorn marketgenius.services.api:app --host 0.0.0.0 --port 8000` |
| `mg-worker` | Celery/RQ задачи (ETL, retrain) | `python -m marketgenius.services.worker` |
| `mg-scheduler` | Prefect/Beat | `prefect agent start -q default` |
| `mg-telegram-pub` | Telegram Publisher bot | `python -m marketgenius.services.telegram_publisher` |
| `mg-telegram-bot` | Telegram Query bot | `python -m marketgenius.services.telegram_query_bot` |

## Мониторинг и логирование

1. **Prometheus + Grafana**: scrape `/metrics` каждого контейнера, dashboards в `infra/grafana/`.
2. **Sentry**: DSN в `.env`, интегрируется в `core.logging`.
3. **Alertmanager**: правила в `infra/alertmanager/alert_rules.yaml`.

## Режимы окружений

| Env | Описание | Особенности |
|-----|----------|-------------|
| `dev` | Локальная разработка | hot-reload, db-volumes на хосте, отключён Sentry |
| `staging` | Пре-прод | nightly retrain, CORS ограничен | 
| `prod` | Продакшн | read-only FS, autoscaling, full observability |

---

_Схема выступает ориентиром. Допустимы отступления при условии соблюдения принципов «разделение ответственности» и «сервисы stateless»._ 