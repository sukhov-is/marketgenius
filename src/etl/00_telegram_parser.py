from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import argparse

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError

# Загрузка переменных окружения
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# Параметры сессии (базовое имя)
session_base_name = 'Load'

# Параметры клиента Telethon
client_kwargs = {
    'device_model': "PC",
    'system_version': "4.16.30-vxCUSTOM",
    'app_version': "7.8.0",
    'lang_code': "en",
    'system_lang_code': "en"
}

# Настройки парсера
BATCH_SIZE = 100  # Размер пакета сообщений
WAIT_TIME = 2     # Задержка между запросами (в секундах)
MAX_CONCURRENT_TASKS = 3  # Максимальное количество одновременных задач
MAX_RETRIES = 3 # Максимальное количество попыток для каждого канала

# Пути к файлам
CONFIG_FILE = "configs/channels_config.json"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"telegram_parser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Создаем директорию для логов, если её нет
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_channels(content_type: str) -> Dict[str, str]:
    """
    Загружает конфигурацию каналов из JSON-файла.
    
    Args:
        content_type (str): Тип контента ('news' или 'blogs')
    
    Returns:
        Dict[str, str]: Словарь, где ключ - ссылка на канал, значение - название канала
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get(content_type, {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Ошибка загрузки конфигурации каналов: {e}")
        return {}

class TelegramParser:
    def __init__(self, session_name: str, api_id: str, api_hash: str, phone: str, content_type: str, last_datetime: datetime | None = None, *, remove_session_file: bool = True):
        """
        Инициализация парсера Telegram.

        Args:
            session_name (str): Имя сессии
            api_id (str): API ID от Telegram
            api_hash (str): API Hash от Telegram
            phone (str): Номер телефона для аутентификации
            content_type (str): Тип контента ('news' или 'blogs')
            last_datetime (datetime | None): Дата последней успешно загруженной новости (используется для инкрементальной выгрузки)
            remove_session_file (bool): Флаг для удаления файла сессии после завершения работы
        """
        self.session_name = session_name
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.content_type = content_type
        self.client = None
        self.messages: List[Dict] = []
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        self.session_file = f"{session_name}.session"  # Добавляем путь к файлу сессии
        self.last_datetime = last_datetime
        self.remove_session_file = remove_session_file

    async def init_client(self) -> None:
        """Инициализация клиента с повторными попытками"""
        for attempt in range(3):
            try:
                self.client = TelegramClient(
                    self.session_name,
                    self.api_id,
                    self.api_hash,
                    **client_kwargs
                )
                await self.client.start(phone=self.phone)
                logger.info("Клиент Telegram успешно инициализирован")
                return
            except Exception as e:
                logger.error(f"Попытка {attempt + 1} подключения не удалась: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(5)

    async def process_channel(self, channel_link: str, channel_name: str) -> None:
        """
        Обработка одного канала с механизмом повторных попыток и адаптивными паузами.

        Args:
            channel_link (str): Ссылка на канал
            channel_name (str): Название канала
        """
        max_retries = MAX_RETRIES
        current_wait_time = WAIT_TIME  # Начальное значение паузы
        consecutive_errors = 0
        
        # Если указана дата последней записи, меняем направление обхода на «сначала новые»
        iterate_reverse = self.last_datetime is None

        for attempt in range(max_retries):
            try:
                logger.info(f"Начало обработки канала: {channel_name}")
                entity = await self.client.get_entity(channel_link)
                message_count = 0
                batch = []
                error_count = 0

                async for message in self.client.iter_messages(
                    entity,
                    limit=None,
                    # Если last_datetime не задана, берём старую логику (от старых к новым).
                    reverse=iterate_reverse
                ):
                    # Пропускаем сообщения, которые уже есть в датасете
                    if self.last_datetime and message.date <= self.last_datetime:
                        # Поскольку мы читаем от новых к старым, можно прервать цикл
                        break
                    if message.message:
                        batch.append({
                            'datetime': message.date,
                            'channel_name': channel_name,
                            'channel_link': channel_link,
                            'message_id': message.id,
                            'news': message.message,
                            'views': getattr(message, 'views', 0)
                        })
                        message_count += 1

                        # Обрабатываем пакет сообщений
                        if len(batch) >= BATCH_SIZE:
                            async with self.semaphore:
                                self.messages.extend(batch)
                                batch = []
                                if message_count % 5000 == 0:
                                    logger.info(f"Обработано {message_count} сообщений из канала {channel_name}")
                                
                                # Адаптивная пауза: уменьшаем время ожидания при успешных запросах
                                if consecutive_errors == 0 and current_wait_time > WAIT_TIME:
                                    current_wait_time = max(WAIT_TIME, current_wait_time * 0.8)
                                    
                                consecutive_errors = 0
                                logger.debug(f"Пауза между пакетами: {current_wait_time:.1f} сек")
                                await asyncio.sleep(current_wait_time)

                # Добавляем оставшиеся сообщения
                if batch:
                    self.messages.extend(batch)

                logger.info(f"Завершена обработка канала {channel_name}. Всего сообщений: {message_count}")
                return

            except FloodWaitError as e:
                wait_time = e.seconds
                logger.warning(f"FloodWaitError в канале {channel_name}: ожидание {wait_time} секунд...")
                await asyncio.sleep(wait_time)
                consecutive_errors += 1
            except asyncio.TimeoutError as e:
                consecutive_errors += 1
                # Увеличиваем время ожидания при последовательных ошибках
                current_wait_time = min(current_wait_time * 1.5, 30)  # Максимум 30 секунд
                logger.warning(f"TimeoutError в канале {channel_name}: повторная попытка через {current_wait_time:.1f} секунд... ({attempt+1}/{max_retries})")
                await asyncio.sleep(current_wait_time)
            except Exception as e:
                consecutive_errors += 1
                current_wait_time = min(current_wait_time * 1.5, 30)
                logger.error(f"Ошибка при обработке канала {channel_name}: {e}")
                if attempt == max_retries - 1:
                    raise
                logger.info(f"Пауза перед повторной попыткой: {current_wait_time:.1f} сек")
                await asyncio.sleep(current_wait_time)

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Подготовка DataFrame с правильным форматированием.
        
        Returns:
            pd.DataFrame: Подготовленный DataFrame с новостями
        """
        if not self.messages:
            return pd.DataFrame()

        df = pd.DataFrame(self.messages)

        # Очистка текста новостей от переносов строк и лишних пробелов
        df['news'] = (df['news']
                    .str.replace('\n', ' ')  # заменяем переносы строк на пробелы
                    .str.replace('\s+', ' ', regex=True)  # заменяем множественные пробелы на один
                    .str.strip())  # удаляем пробелы в начале и конце

        # Создание ссылок на сообщения
        df['message_link'] = df.apply(
            lambda row: f"https://{row['channel_link']}/{row['message_id']}"
            if row['channel_link'].startswith("t.me")
            else f"{row['channel_link']}/{row['message_id']}",
            axis=1
        )

        # Очистка и сортировка DataFrame
        df.drop(columns=['channel_link', 'message_id'], inplace=True)
        df = df[['datetime', 'channel_name', 'message_link', 'views', 'news']] 
        df.sort_values(by='datetime', inplace=True)

        return df

    async def cleanup(self) -> None:
        """Очистка ресурсов и удаление файла сессии"""
        if self.client:
            await self.client.disconnect()
            logger.info("Клиент Telegram отключен")
        
        if self.remove_session_file and os.path.exists(self.session_file):
            try:
                os.remove(self.session_file)
                logger.info(f"Файл сессии {self.session_file} удален")
            except Exception as e:
                logger.error(f"Ошибка при удалении файла сессии: {e}")

    async def run(self) -> None:
        """Основной метод запуска парсера"""
        try:
            await self.init_client()
            channels = load_channels(self.content_type)

            if not channels:
                logger.error("Не найдены каналы для обработки")
                return

            logger.info(f"Начало обработки {len(channels)} каналов типа {self.content_type}")

            # Создаем пул задач с ограничением количества одновременных задач
            tasks = []
            for channel_link, channel_name in channels.items():
                task = asyncio.create_task(self.process_channel(channel_link, channel_name))
                tasks.append(task)
                await asyncio.sleep(1)  # Пауза между запуском обработки каналов

            # Запускаем задачи
            await asyncio.gather(*tasks, return_exceptions=True)

            # Сохранение результатов
            df = self._prepare_dataframe()
            if not df.empty:
                filename = "blogs.csv" if self.content_type == "blogs" else "telegram_news.csv"
                output_path = f"data/external/news_tg_csv/{filename}"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"Данные сохранены в файле {output_path}")
            else:
                logger.warning("Нет данных для сохранения")

        except Exception as e:
            logger.error(f"Критическая ошибка при выполнении парсера: {e}")
            raise

        finally:
            await self.cleanup()  # Заменяем старый код отключения на новый метод cleanup

async def main():
    """Точка входа в программу"""
    parser = argparse.ArgumentParser(description="Парсер Telegram каналов")
    parser.add_argument(
        "--content-type",
        type=str,
        choices=["news", "blogs", "all"],
        help="Тип контента для загрузки (news, blogs или all)"
    )
    args = parser.parse_args()

    content_type = args.content_type
    if not content_type:
        while True:
            user_input = input("Выберите тип контента для загрузки (n/b/a - news/blogs/all): ").lower()
            if user_input in ['n', 'b', 'a']:
                content_type = {'n': 'news', 'b': 'blogs', 'a': 'all'}[user_input]
                break
            print("Неверный ввод. Пожалуйста, используйте n, b или a.")

    try:
        if content_type == "all":
            # Одна общая сессия для обоих типов контента
            shared_session = f"{session_base_name}_all_{datetime.now().strftime('%H%M%S')}"

            news_parser = TelegramParser(
                session_name=shared_session,
                api_id=API_ID,
                api_hash=API_HASH,
                phone=PHONE,
                content_type="news",
                remove_session_file=False,  # Не удаляем, будет использоваться повторно
            )
            await news_parser.run()

            blogs_parser = TelegramParser(
                session_name=shared_session,
                api_id=API_ID,
                api_hash=API_HASH,
                phone=PHONE,
                content_type="blogs",
                remove_session_file=True,  # После блогов удаляем
            )
            await blogs_parser.run()

        else:
            # Обработка одного типа контента, сессия удаляется после завершения
            single_session = f"{session_base_name}_{content_type}_{datetime.now().strftime('%H%M%S')}"
            single_parser = TelegramParser(
                session_name=single_session,
                api_id=API_ID,
                api_hash=API_HASH,
                phone=PHONE,
                content_type=content_type,
            )
            await single_parser.run()

    except Exception as e:
        logger.critical(f"Программа завершилась с ошибкой: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())