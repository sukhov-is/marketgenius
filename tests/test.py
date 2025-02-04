import re
import emoji
import time

# Тестовые строки
short_text = "Привет! 😊 Как дела? 🚀"
long_text = "Это тестовый текст с разными эмодзи 😊🔥🎉💖👍💡🚀💯⚽🎶🌟🐱‍👤🤖👾💀👽 и еще больше эмодзи " * 100

# Регулярное выражение для удаления эмодзи
EMOJI_PATTERN = re.compile(
    "["   
    "\U0001F1E0-\U0001F1FF"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F600-\U0001F64F"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F700-\U0001F77F"  
    "\U0001F780-\U0001F7FF"  
    "\U0001F800-\U0001F8FF"  
    "\U0001F900-\U0001F9FF"  
    "\U0001FA00-\U0001FA6F"  
    "\U0001FA70-\U0001FAFF"  
    "\U00002600-\U00002B55"
    "\U0000FE0E-\U0000FE0F"
    "]", 
    flags=re.UNICODE
)

# Функции удаления эмодзи
def remove_emoji_re(text: str) -> str:
    return EMOJI_PATTERN.sub(r"", text)

def remove_emoji_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace="")

# Тестирование скорости выполнения
def measure_time(func, text, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        func(text)
    return time.time() - start_time

# Запускаем тесты
results = {
    "Regex (short text)": measure_time(remove_emoji_re, short_text, 10000),
    "Emoji (short text)": measure_time(remove_emoji_emoji, short_text, 10000),
    "Regex (long text)": measure_time(remove_emoji_re, long_text, 100),
    "Emoji (long text)": measure_time(remove_emoji_emoji, long_text, 100),
}

# Вывод результатов
print("\nРезультаты тестирования скорости удаления эмодзи:")
for method, time_taken in results.items():
    print(f"{method}: {time_taken:.6f} секунд")
