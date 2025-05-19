from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import Dict, Any, List

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает все источники. Для продакшена лучше указать конкретные домены.
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все методы (GET, POST и т.д.)
    allow_headers=["*"],  # Разрешает все заголовки
)

# Путь к папке с данными
DATA_DIR = "/c:/Users/Admin/Documents/MarketGenius/data/features_final"

# Список полей, которые будем возвращать
SELECTED_FIELDS = [
    "date", "SECID", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "RSI",
    "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "NetProfit_y",
    "Revenue_y", "PE_y", "PB_y", "WeightedIndices_blog_score_roll_avg_30",
    "WeightedIndices_news_score_roll_avg_30"
]

@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str) -> Dict[str, Any]:
    file_path = os.path.join(DATA_DIR, f"{ticker.upper()}_final.csv")
    if not os.path.exists(file_path):
        return {"error": "Ticker not found"}

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return {"error": "No data available for this ticker"}
        
        # Берем последнюю строчку
        latest_data = df.iloc[-1]
        
        # Выбираем только нужные поля
        result_data = latest_data[SELECTED_FIELDS].to_dict()
        
        return result_data
    except Exception as e:
        return {"error": str(e)}

# Для запуска приложения локально:
# uvicorn main:app --reload 