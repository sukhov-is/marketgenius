import asyncio
import json
import os
import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError

# Загрузка переменных окружения
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# Параметры сессии
session_name = 'Load_blogs'  

# Параметры клиента Telethon
client_kwargs = {
    'device_model': "PC",                     
    'system_version': "4.16.30-vxCUSTOM",       
    'app_version': "7.8.0",                     
    'lang_code': "en",                          
    'system_lang_code': "en"                    
}

# Загрузка конфигурации каналов из JSON-файла
CONFIG_FILE = "channels_config.json"

def load_channels():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("blogs", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки конфигурации каналов: {e}")
        return []

async def main():
    # Инициализация клиента Telethon
    client = TelegramClient(session_name, API_ID, API_HASH, **client_kwargs)
    await client.start(phone=PHONE)

    all_messages = []  # Список для хранения всех сообщений
    channels = load_channels()

    for channel in channels:
        print(f"Получаем сообщения из канала: {channel}")
        try:
            entity = await client.get_entity(channel)
            async for message in client.iter_messages(entity, limit=None, reverse=True):
                if message.message:
                    dt = message.date  # объект datetime
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M:%S")
                    all_messages.append({
                        'date': date_str,
                        'time': time_str,
                        'channel': channel,
                        'news': message.message  # Оригинальный текст без очистки
                    })
        except FloodWaitError as e:
            print(f"FloodWaitError: Ждем {e.seconds} секунд...")
            await asyncio.sleep(e.seconds)
        except Exception as e:
            print(f"Ошибка при обработке канала {channel}: {e}")

    # Создание DataFrame и сохранение в CSV
    df = pd.DataFrame(all_messages)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.sort_values(by='datetime', inplace=True)
        df.drop(columns=['datetime'], inplace=True)
        df.to_csv("data/external/news_tg_csv/blogs.csv", index=False)
        print("Данные сохранены в файле blogs.csv")
    else:
        print("Нет данных для сохранения.")

    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
