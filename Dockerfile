# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в рабочую директорию
COPY . .

# Указываем команду для запуска приложения
# Запускаем Uvicorn, чтобы он был доступен извне контейнера на порту 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 