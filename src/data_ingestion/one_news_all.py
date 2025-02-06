import asyncio
import re
import pandas as pd
import emoji
import os
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError

# Загрузка переменных окружения из .env
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# Название сессии 
session_name = 'Load_news'

# Параметры клиента
client_kwargs = {
    'device_model': "PC",
    'system_version': "4.16.30-vxCUSTOM",
    'app_version': "7.8.0",
    'lang_code': "en",
    'system_lang_code': "en"
}

# Функция очистки текста (удаляет ссылки, эмодзи и лишние пробелы)
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)          # удаляем ссылки
    text = emoji.replace_emoji(text, replace='')  # удаляем эмодзи
    text = re.sub(r'\s+', ' ', text).strip()        # удаляем лишние пробелы
    return text

# Функция для загрузки истории сообщений из одного канала
async def fetch_channel_history(client, channel):
    all_messages = []
    try:
        entity = await client.get_entity(channel)
        # Загружаем все сообщения (начиная с самых старых)
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
    return all_messages

async def main():
    # Задайте имя нового канала:
    new_channel = 't.me/marketsnapshot'  

    # Путь к CSV-файлу (тот же, что используется в основном скрипте)
    csv_file = "data/external/news_tg_csv/telegram_news.csv"
    
    # Проверяем, существует ли CSV-файл
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        # Если канал уже присутствует в файле, выходим, чтобы не перегружать его повторно
        if new_channel in df_existing['channel'].unique():
            print(f"Канал {new_channel} уже загружен в файл.")
            return
    else:
        print(f"Файл {csv_file} не найден. Будет создан новый.")
        df_existing = pd.DataFrame(columns=['date', 'time', 'channel', 'news'])

    # Инициализируем Telethon-клиент
    client = TelegramClient(session_name, API_ID, API_HASH, **client_kwargs)
    await client.start(phone=PHONE)

    print(f"Получаем сообщения из канала: {new_channel}")
    new_messages = await fetch_channel_history(client, new_channel)
    await client.disconnect()

    if not new_messages:
        print("Не удалось получить сообщения или сообщений нет.")
        return

    # Создаем DataFrame для новых сообщений
    df_new = pd.DataFrame(new_messages)

    # Объединяем с уже существующими данными
    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    # Сортируем данные по дате и времени.
    # Для этого создаем временный столбец datetime
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'])
    df_all.sort_values(by='datetime', inplace=True)
    df_all.drop(columns=['datetime'], inplace=True)

    # Сохраняем обновленный CSV-файл
    df_all.to_csv(csv_file, index=False)
    print(f"Файл обновлен и сохранен: {csv_file}")

if __name__ == '__main__':
    asyncio.run(main())
