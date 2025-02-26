import re
from typing import Any, Dict

# Регулярное выражение для поиска эмодзи в тексте
EMOJI_PATTERN = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # Флаги (iOS)
    "\U0001F300-\U0001F5FF"  # Символы и пиктограммы
    "\U0001F600-\U0001F64F"  # Эмоции
    "\U0001F680-\U0001F6FF"  # Транспорт и карты
    "\U0001F700-\U0001F77F"  # Алхимические символы
    "\U0001F780-\U0001F7FF"  # Геометрические фигуры
    "\U0001F800-\U0001F8FF"  # Дополнительные стрелки
    "\U0001F900-\U0001F9FF"  # Дополнительные символы
    "\U0001FA00-\U0001FA6F"  # Шахматные символы
    "\U0001FA70-\U0001FAFF"  # Расширенные символы
    "\U00002300-\U00002BFF"  # Технические символы
    "\U0000FE0E-\U0000FE0F"  # Варианты отображения символов
    "])",
    flags=re.UNICODE,
)

# Регулярные выражения для удаления URL, упоминаний и хэштегов
URL_PATTERN = re.compile(r"(http\S+|www\.\S+)", flags=re.UNICODE)
URL_WITHOUT_HTTP_PATTERN = re.compile(r"[\S]+\.(ru|me|com|org)[/][\S]+", flags=re.UNICODE)
USERS_PATTERN = re.compile(r"\s@(\w+)", flags=re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#(\w+)", flags=re.UNICODE)

# Функция для удаления эмодзи из текста
def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub(r"", text)

# Функция для удаления хэштегов
def remove_hashtags(text: str) -> str:
    return HASHTAG_PATTERN.sub(r"", text)

# Функция для удаления ссылок
def remove_urls(text: str) -> str:
    text1 = URL_PATTERN.sub(r"", text)
    return URL_WITHOUT_HTTP_PATTERN.sub(r"", text1)

# Функция для удаления упоминаний пользователей
def remove_users(text: str) -> str:
    return USERS_PATTERN.sub(r"", text)

# Функция для исправления абзацев (удаление лишних пробелов и пустых строк)
def fix_paragraphs(text: str) -> str:
    paragraphs = text.split("\n")
    for i, paragraph in enumerate(paragraphs):
        paragraphs[i] = " ".join(paragraph.split()).strip()
    paragraphs = [p for p in paragraphs if len(p) >= 3]  # Удаляем слишком короткие абзацы
    return "\n".join(paragraphs)

# Функция для исправления пунктуации
def remove_bad_punct(text: str) -> str:
    paragraphs = text.split("\n")
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.replace(". .", ".").replace("..", ".")
        paragraph = paragraph.replace("« ", "«").replace(" »", "»")
        paragraph = paragraph.replace(" :", ":")
        paragraph = paragraph.replace("\xa0", " ")  # Убираем неразрывные пробелы
        paragraphs[i] = paragraph
    return "\n".join(paragraphs)

# Класс для обработки текста
class TextProcessor:
    def __init__(self, config: Dict[str, Any]) -> None:
        # Последовательность функций обработки текста
        self.pipeline = (
            remove_emoji,
            remove_hashtags,
            remove_users,
            remove_urls,
            remove_bad_punct,
            fix_paragraphs,
        )
        # Список подстрок, которые приводят к полной фильтрации текста
        self.skip_substrings = config["skip_substrings"]
        # Список подстрок, которые будут заменены на пробел
        self.rm_substrings = config["rm_substrings"]
        # Список запрещенных слов
        self.obscene_substrings = config["obscene_substrings"]
        # Список ключевых слов для предварительной фильтрации
        self.filter_words = config.get("filter_words", [])

    def __call__(self, text: str) -> str:
        if not text:
            return ""
            
        # Предварительная фильтрация текста по ключевым словам и символам
        if self.prefilter_text(text):
            return ""
            
        if self.is_bad_text(text):  # Если текст содержит запрещенные слова, игнорируем его
            return ""
        text = self.remove_bad_text(text)  # Удаляем нежелательные подстроки
        
        # Применяем все функции обработки текста
        for step in self.pipeline:
            text = step(text)
        
        if self.is_bad_text(text):  # Повторно проверяем текст
            return ""
        text = self.remove_bad_text(text)
        
        return text.strip()

    # Функция для предварительной фильтрации текста
    def prefilter_text(self, text: str) -> bool:
        """
        Проверяет наличие ключевых слов или символов в тексте.
        Возвращает True, если текст нужно фильтровать.
        """
        return any(filter_word in text.lower() for filter_word in self.filter_words)

    # Функция для проверки наличия запрещенных слов в тексте
    def has_obscene(self, text: str) -> bool:
        return any(ss in text for ss in self.obscene_substrings)

    # Функция для проверки, содержит ли текст запрещенные подстроки
    def is_bad_text(self, text: str) -> bool:
        return any(ss in text for ss in self.skip_substrings)

    # Функция для удаления запрещенных подстрок
    def remove_bad_text(self, text: str) -> str:
        for ss in self.rm_substrings:
            if ss in text:
                text = text.replace(ss, " ")
        return text