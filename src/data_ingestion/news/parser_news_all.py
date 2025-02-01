import asyncio
import re
import pandas as pd
import emoji
import os
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
# from config import API_ID, API_HASH, PHONE

# Загрузка переменных окружения
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# Параметры сессии
session_name = 'Load_news'  

# # Создание Telegram клиента
client_kwargs = {
    'device_model': "PC",                     
    'system_version': "4.16.30-vxCUSTOM",       
    'app_version': "7.8.0",                     
    'lang_code': "en",                          
    'system_lang_code': "en"                    
}

# Список каналов (username или ссылка)
channels = [
    't.me/banksta',
    't.me/newssmartlab',
    't.me/markettwits',
    't.me/bcs_express',
    't.me/ifax_go',
    't.me/Information_disclosure',
    't.me/bbbreaking',
    't.me/tass_agency',
    't.me/interfaxonline'
    # добавьте необходимые каналы
]

def clean_text(text):
    """
    Функция очистки текста:
      - удаляет ссылки,
      - убирает эмодзи,
      - удаляет лишние пробелы и переносы строк.
    """
    if not text:
        return ""
    # Удаляем ссылки (http, https)
    text = re.sub(r'http\S+', '', text)
    # Удаляем эмодзи
    text = emoji.replace_emoji(text, replace='')
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def main():
    # Инициализация клиента Telethon
    client = TelegramClient(session_name, API_ID, API_HASH, **client_kwargs)
    await client.start(phone=PHONE)

    all_messages = []  # список для хранения всех сообщений

    for channel in channels:
        print(f"Получаем сообщения из канала: {channel}")
        try:
            # Получаем сущность канала
            entity = await client.get_entity(channel)
            # Перебор всех сообщений канала.
            # Параметр reverse=True позволяет начать с самых старых сообщений.
            async for message in client.iter_messages(entity, limit=None, reverse=True):
                if message.message:
                    dt = message.date  # объект datetime
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M:%S")
                    cleaned_text = clean_text(message.message)
                    all_messages.append({
                        'date': date_str,
                        'time': time_str,
                        'channel': channel,
                        'news': cleaned_text
                    })
        except FloodWaitError as e:
            print(f"FloodWaitError: Ждем {e.seconds} секунд...")
            await asyncio.sleep(e.seconds)
        except Exception as e:
            print(f"Ошибка при обработке канала {channel}: {e}")

    # Создаем DataFrame и сортируем сообщения по дате и времени
    df = pd.DataFrame(all_messages)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.sort_values(by='datetime', inplace=True)
        df.drop(columns=['datetime'], inplace=True)
        # Сохраняем данные в CSV-файл
        df.to_csv("data/external/news_tg_csv/telegram_news.csv", index=False)
        print("Данные сохранены в файле telegram_news.csv")
    else:
        print("Нет данных для сохранения.")

    # disconnect() завершает только текущее соединение, не вызывайте log_out()
    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())