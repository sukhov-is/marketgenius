from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Глобальные настройки DAG
# ---------------------------------------------------------------------------

# ПРЕДПОЛОЖЕНИЕ: Корень вашего проекта. Измените, если необходимо.
# Этот путь должен быть доступен из окружения, где запускается Airflow.
PROJECT_ROOT = '/c:/Users/Admin/Documents/MarketGenius' # Используем прямой слеш

# Пути к конфигурационным файлам (если они не передаются как аргументы в каждый скрипт)
# Убедитесь, что эти пути корректны и доступны из PROJECT_ROOT
DEFAULT_COMPANIES_CONFIG = f'{PROJECT_ROOT}/configs/all_companies_config.json'
# DEFAULT_INDICES_CONFIG = f'{PROJECT_ROOT}/configs/indices_config.json' # Раскомментируйте, если используется в 01
DEFAULT_DAILY_SUMMARY_CONFIG = f'{PROJECT_ROOT}/configs/daily_summary_batch_config.json'

# Директории для данных (для справки и для BashOperator команд, если нужно)
DATA_RAW_DIR = f'{PROJECT_ROOT}/data/raw'
DATA_PROCESSED_DIR = f'{PROJECT_ROOT}/data/processed'
DATA_FEATURES_FINAL_DIR = f'{PROJECT_ROOT}/data/features_final'
DATA_BATCH_REQUESTS_DIR = f'{PROJECT_ROOT}/data/batch_requests'
TELEGRAM_SUMMARY_DIR = f'{PROJECT_ROOT}/data/processed/telegram_daily_summary'
GPT_RESULTS_DIR = f'{PROJECT_ROOT}/data/processed/gpt'

# Абсолютные пути к скриптам
SCRIPT_01_PATH = f'{PROJECT_ROOT}/scripts/01_run_daily_update.py'
SCRIPT_02_PATH = f'{PROJECT_ROOT}/scripts/02_run_processing_pipeline.py' # Используется ли этот скрипт?
SCRIPT_03_PATH = f'{PROJECT_ROOT}/scripts/03_update_telegram_pipeline.py'
SCRIPT_04_PATH = f'{PROJECT_ROOT}/scripts/04_run_parallel_batches.py'
SCRIPT_05_PATH = f'{PROJECT_ROOT}/scripts/05_filling_estimates.py'
SCRIPT_06_PATH = f'{PROJECT_ROOT}/scripts/06_process_targets.py'
SCRIPT_07_PATH = f'{PROJECT_ROOT}/scripts/07_prepare_daily_summary_batch_file.py'
SCRIPT_08_PATH = f'{PROJECT_ROOT}/scripts/08_process_daily_summary_batches.py'
SCRIPT_09_PATH = f'{PROJECT_ROOT}/scripts/09_publish_pipeline.py'

# ---------------------------------------------------------------------------
# Определение DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id='market_genius_full_pipeline',
    start_date=pendulum.datetime(2023, 10, 27, tz="UTC"), # Установите актуальную дату начала
    catchup=False,
    schedule='@daily', # Ежедневный запуск (например, в полночь UTC)
    tags=['market_genius', 'data_pipeline'],
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': pendulum.duration(minutes=5),
    },
    doc_md="""
    ## Комплексный пайплайн Market Genius

    Этот DAG автоматизирует полный цикл сбора, обработки, анализа и публикации рыночных данных и новостей.

    **Основные этапы:**
    1.  **Ежедневное обновление данных**: Загрузка сырых рыночных данных.
    2.  **Конвейер обработки данных**: Обработка рыночных данных (технические индикаторы, фин. отчеты и т.д.).
        *Примечание: уточните, используется ли этот шаг и какие скрипты он запускает.*
    3.  **Обновление данных из Telegram**: Загрузка и первичная обработка новостей/блогов из Telegram.
    4.  **Параллельная обработка батчей (GPT)**: Глубокая обработка текстов новостей/блогов (вероятно, с помощью GPT).
    5.  **Заполнение оценок**: Генерация признаков на основе GPT-оценок и данных тикеров.
    6.  **Обработка целевых переменных**: Расчет таргет-колонок для моделей.
    7.  **Подготовка батч-файла для дневных саммари**: Создание запросов для OpenAI API для генерации дневных саммари.
    8.  **Обработка батчей дневных саммари**: Отправка запросов в OpenAI, получение и сохранение дневных саммари.
    9.  **Подготовка данных для публикации**: Копирование/переименование файлов с дневными саммари в ожидаемые для публикации пути.
    10. **Публикация в Telegram**: Отправка сгенерированных сообщений в Telegram.

    **Переменные Airflow (необходимо создать):**
    - `telegram_api_id`
    - `telegram_api_hash`
    - `telegram_phone`
    - `openai_api_key`
    - `telegram_bot_token`
    - `telegram_chat_id`
    """
) as dag:

    # --- Задача 1: Ежедневное обновление данных ---
    task_01_run_daily_update = BashOperator(
        task_id='run_daily_update_01',
        bash_command=(
            f'python {SCRIPT_01_PATH} '
            f'--output-dir {DATA_RAW_DIR} '
            f'--companies-config {DEFAULT_COMPANIES_CONFIG} '
            # f'--indices-config {DEFAULT_INDICES_CONFIG} ' # Раскомментируйте, если используется
            '--log-level INFO'
        ),
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 1: Ежедневное обновление данных
        Запускает `01_run_daily_update.py` для сбора последних рыночных данных.
        """
    )

    # --- Задача 2: Конвейер обработки данных (УТОЧНИТЬ НЕОБХОДИМОСТЬ И АРГУМЕНТЫ) ---
    task_02_run_processing_pipeline = BashOperator(
        task_id='run_processing_pipeline_02',
        bash_command=(
            f'python {SCRIPT_02_PATH} '
            # По умолчанию скрипт 02 использует 'scripts' как --scripts-dir.
            # Если его под-скрипты (technicalIndicators.py и т.д.) лежат в src/data_ingestion/,
            # то команда может быть такой. Либо адаптируйте скрипт 02 или его вызов.
            f'--scripts-dir {PROJECT_ROOT}/src/data_ingestion '
            '--log-level INFO'
        ),
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 2: Запуск конвейера обработки данных
        Оркестрирует выполнение нескольких под-скриптов обработки данных (`technicalIndicators.py`, и т.д.).
        **ВАЖНО**: Убедитесь, что этот шаг необходим, и что пути к под-скриптам и их аргументы корректны.
        Если этот шаг не нужен, удалите эту задачу и скорректируйте зависимости.
        Предполагается, что этот шаг готовит данные в `data/processed/ready_for_training/`
        и, возможно, `data/processed/tickers_indices.csv`.
        """
    )

    # --- Задача 3: Обновление данных из Telegram ---
    task_03_update_telegram = BashOperator(
        task_id='update_telegram_pipeline_03',
        bash_command=(
            f'python {SCRIPT_03_PATH} '
            '--content-type all '
        ),
        cwd=PROJECT_ROOT,
        env={ 
            'API_ID': '{{ var.value.telegram_api_id }}',
            'API_HASH': '{{ var.value.telegram_api_hash }}',
            'PHONE': '{{ var.value.telegram_phone }}'
        },
        doc_md="""
        #### Задача 3: Обновление данных из Telegram
        Запускает `03_update_telegram_pipeline.py` для загрузки и обработки сообщений из Telegram.
        """
    )

    # --- Задача 4: Параллельная обработка батчей (GPT) ---
    task_04_run_parallel_batches = BashOperator(
        task_id='run_parallel_batches_04',
        bash_command=(
            f'python {SCRIPT_04_PATH} '
            '--log-level INFO'
        ),
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 4: Параллельная обработка батчей (новости/блоги)
        Запускает `04_run_parallel_batches.py` для глубокой обработки текстов.
        Результаты включают `gpt_blogs_history.jsonl`, `gpt_news_history.jsonl`
        и `results_gpt_blogs.csv`, `results_gpt_news.csv` в `data/processed/gpt/`.
        """
    )

    # --- Задача 5: Заполнение оценок ---
    task_05_filling_estimates = BashOperator(
        task_id='filling_estimates_05',
        bash_command=f'python {SCRIPT_05_PATH}',
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 5: Заполнение оценок и генерация признаков
        Запускает `05_filling_estimates.py`. Читает GPT-оценки и данные тикеров,
        сохраняет результат в `data/features_final/`.
        """
    )

    # --- Задача 6: Обработка целевых переменных ---
    task_06_process_targets = BashOperator(
        task_id='process_targets_06',
        bash_command=f'python {SCRIPT_06_PATH}',
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 6: Обработка целевых переменных
        Запускает `06_process_targets.py`. Обновляет файлы в `data/features_final/` таргетами.
        """
    )

    # --- Задача 7: Подготовка батч-файла для дневных саммари ---
    task_07_prepare_daily_summary_batch = BashOperator(
        task_id='prepare_daily_summary_batch_file_07',
        bash_command=(
            f'python {SCRIPT_07_PATH} '
            f'--config-file {DEFAULT_DAILY_SUMMARY_CONFIG} '
            '--log-level INFO'
        ),
        cwd=PROJECT_ROOT,
        doc_md="""
        #### Задача 7: Подготовка батч-файла для дневных саммари
        Запускает `07_prepare_daily_summary_batch_file.py` для создания JSONL запросов
        в `data/batch_requests/`.
        """
    )

    # --- Задача 8: Обработка батчей дневных саммари ---
    task_08_process_daily_summary_batches = BashOperator(
        task_id='process_daily_summary_batches_08',
        bash_command=(
            f'python {SCRIPT_08_PATH} '
            '--source all '
            f'--input-dir {DATA_BATCH_REQUESTS_DIR} '
            f'--output-dir {TELEGRAM_SUMMARY_DIR} '
            f'--gpt-results-dir {GPT_RESULTS_DIR} '
            '--log-level INFO '
            '--threads True'
        ),
        cwd=PROJECT_ROOT,
        env={'OPENAI_API_KEY': '{{ var.value.openai_api_key }}'},
        doc_md="""
        #### Задача 8: Обработка батчей дневных саммари
        Запускает `08_process_daily_summary_batches.py` для получения дневных саммари от OpenAI.
        Сохраняет результаты в `data/processed/telegram_daily_summary/`.
        """
    )
    
    # --- Задача 9a: Подготовка данных для публикации (копирование/переименование) ---
    # Выходные файлы из task_08:
    # data/processed/telegram_daily_summary/daily_summary_blogs_telegram.csv
    # data/processed/telegram_daily_summary/daily_summary_news_telegram.csv
    # Ожидаемые task_09:
    # data/processed/gpt/telegram_blogs.csv
    # data/processed/gpt/telegram_news.csv
    
    # Убедимся, что целевая директория для task_09 существует
    ensure_gpt_dir_command = f'mkdir -p {GPT_RESULTS_DIR}'
    
    copy_blogs_command = f'cp {TELEGRAM_SUMMARY_DIR}/daily_summary_blogs_telegram.csv {GPT_RESULTS_DIR}/telegram_blogs.csv'
    copy_news_command = f'cp {TELEGRAM_SUMMARY_DIR}/daily_summary_news_telegram.csv {GPT_RESULTS_DIR}/telegram_news.csv'
    
    # В Windows PowerShell команда cp - это alias для Copy-Item. mkdir -p тоже должно работать.
    # Для кросс-платформенности можно использовать && для последовательного выполнения.
    # Если cp или mkdir не работают в вашем PowerShell окружении Airflow, можно использовать PythonOperator
    # или изменить пути в скриптах 08 или 09.
    
    task_09a_prepare_publish_data = BashOperator(
        task_id='prepare_publish_data_09a',
        bash_command=f'{ensure_gpt_dir_command} && {copy_blogs_command} && {copy_news_command}',
        cwd=PROJECT_ROOT, # cwd не так важен для абсолютных путей, но пусть будет
        doc_md="""
        #### Задача 9a: Подготовка данных для публикации
        Копирует/переименовывает файлы результатов из task_08 (`telegram_daily_summary/*_telegram.csv`)
        в пути, ожидаемые task_09 (`gpt/telegram_*.csv`).
        Создает директорию `data/processed/gpt/`, если она не существует.
        """
    )

    # --- Задача 9b: Публикация в Telegram ---
    task_09b_publish_pipeline = BashOperator(
        task_id='publish_pipeline_09b',
        bash_command=f'python {SCRIPT_09_PATH}',
        cwd=PROJECT_ROOT,
        env={
            'TG_BOT_TOKEN': '{{ var.value.telegram_bot_token }}',
            'TG_CHAT_ID': '{{ var.value.telegram_chat_id }}'
        },
        doc_md="""
        #### Задача 9b: Публикация результатов в Telegram
        Запускает `09_publish_pipeline.py`.
        """
    )

    # ---------------------------------------------------------------------------
    # Определение зависимостей (ПРЕДПОЛОЖИТЕЛЬНЫЕ - ТРЕБУЮТ ВАШЕЙ ПРОВЕРКИ)
    # ---------------------------------------------------------------------------

    # Базовая загрузка данных
    # task_01_run_daily_update -> ... (данные из 01 нужны для 02 и/или 05)

    # Ветка обработки Telegram данных и GPT-анализа текстов
    task_03_update_telegram >> task_04_run_parallel_batches

    # Ветка основной обработки данных (если task_02 используется)
    # Если task_02 не используется, то task_05 будет зависеть напрямую от task_01
    # и, возможно, task_04 (если GPT оценки нужны до заполнения других оценок)
    task_01_run_daily_update >> task_02_run_processing_pipeline
    
    # task_05 зависит от результатов GPT (task_04) и обработанных данных (task_02 или task_01)
    [task_02_run_processing_pipeline, task_04_run_parallel_batches] >> task_05_filling_estimates
    # Если task_02 не используется:
    # [task_01_run_daily_update, task_04_run_parallel_batches] >> task_05_filling_estimates

    task_05_filling_estimates >> task_06_process_targets

    # Подготовка и обработка дневных саммари
    # task_07 использует результаты task_04 (gpt_*.history.jsonl)
    task_04_run_parallel_batches >> task_07_prepare_daily_summary_batch
    
    # task_08 использует результаты task_07 (батч-файлы) и task_04 (results_gpt_*.csv)
    [task_07_prepare_daily_summary_batch, task_04_run_parallel_batches] >> task_08_process_daily_summary_batches

    # Публикация
    task_08_process_daily_summary_batches >> task_09a_prepare_publish_data
    task_09a_prepare_publish_data >> task_09b_publish_pipeline

    # Общая зависимость: если какие-то финальные данные из task_06 (features_final)
    # нужны для публикации или дневных саммари, то это тоже нужно учесть.
    # Например, если task_09b_publish_pipeline или task_07/08 косвенно используют
    # результаты из data/features_final/, то:
    # task_06_process_targets >> task_07_prepare_daily_summary_batch (или task_08, или task_09a)

    # Пожалуйста, раскомментируйте и настройте правильные цепочки зависимостей ниже,
    # удалив или закомментировав те, которые не соответствуют вашей логике.

    # ПРИМЕР ПОЛНОЙ ЦЕПОЧКИ (С УЧЕТОМ ПРЕДПОЛОЖЕНИЙ):
    # task_01_run_daily_update >> task_02_run_processing_pipeline
    # task_03_update_telegram >> task_04_run_parallel_batches
    
    # [task_02_run_processing_pipeline, task_04_run_parallel_batches] >> task_05_filling_estimates
    # task_05_filling_estimates >> task_06_process_targets
    
    # # task_07 зависит от task_04
    # # task_08 зависит от task_07 и task_04 (повторная зависимость от task_04 учтется Airflow)
    # task_04_run_parallel_batches >> task_07_prepare_daily_summary_batch >> task_08_process_daily_summary_batches
    
    # # Финальная публикация
    # task_08_process_daily_summary_batches >> task_09a_prepare_publish_data >> task_09b_publish_pipeline

    # # Если данные из task_06 нужны для финальной стадии перед публикацией (например, для task_07 или task_08)
    # # то task_06_process_targets должен быть перед ними. Сейчас он параллелен ветке Telegram саммари.
    # # Если они независимы до самого конца, то можно оставить так или объединить их на task_09b, если это имеет смысл.

    # # Более линейный вариант, если предположить, что все шаги последовательны
    # # и результаты предыдущих нужны последующим:
    # task_01_run_daily_update >> task_02_run_processing_pipeline
    # task_02_run_processing_pipeline >> task_03_update_telegram # Если Telegram обработка зависит от предыдущих
    # task_03_update_telegram >> task_04_run_parallel_batches
    # task_04_run_parallel_batches >> task_05_filling_estimates # task_02 уже был выше
    # task_05_filling_estimates >> task_06_process_targets
    # task_06_process_targets >> task_07_prepare_daily_summary_batch # Если саммари для GPT нужны финальные признаки
    # task_07_prepare_daily_summary_batch >> task_08_process_daily_summary_batches
    # task_08_process_daily_summary_batches >> task_09a_prepare_publish_data >> task_09b_publish_pipeline

