from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai

# Попробуем импортировать tiktoken для более точного подсчёта токенов.
try:
    import tiktoken
except ImportError:  # библиотека необязательна, работаем без неё
    tiktoken = None  # type: ignore

_logger = logging.getLogger(__name__)


class GPTNewsAnalyzer:
    """Универсальный анализатор новостей/сообщений через ChatGPT.

    Пример использования::

        analyzer = GPTNewsAnalyzer()
        df = analyzer.analyze_csv(
            path="data/news.csv",
            date_col="date",
            text_col="text",
            title_col="channel_name",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_csv="data/results.csv",
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        prompt_path: str | os.PathLike = "src/prompts",
        config_path: str | os.PathLike = "configs/companies_config.json",
        max_tokens: int = 8_000,  # лимит на prompt+response
        chunk_limit: int = 50,  # максимум сообщений в одном запросе
        retries: int = 6,
    ) -> None:
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY не задан")

        self.model = model
        self.max_tokens = max_tokens
        self.chunk_limit = chunk_limit
        # prompt_path может быть как директория с несколькими шаблонами,
        # так и конкретным файлом. Если путь — директория, ищем news & blog.
        pp = Path(prompt_path)
        if pp.is_dir():
            self._prompts = {
                f.stem.replace("_promt", ""): f.read_text(encoding="utf-8")
                for f in pp.glob("*_promt.txt")
            }
            if not self._prompts:
                raise FileNotFoundError(f"Не найдены файлы *_promt.txt в {prompt_path}")
        else:
            prompt_name = pp.stem.replace("_promt", "")
            self._prompts = {prompt_name: pp.read_text(encoding="utf-8")}

        self.tickers_block = self._load_tickers(config_path)
        self._retries = retries

    # ------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------
    def analyze_csv(
        self,
        path: str | os.PathLike,
        date_col: str = "datetime",
        text_col: str = "news",
        title_col: str = "channel_name",
        start_date: str | None = None,
        end_date: str | None = None,
        output_csv: str | os.PathLike | None = None,
        prompt_type: str = "news",
    ) -> pd.DataFrame:
        """Анализирует CSV/TSV файл и возвращает DataFrame с результатами."""
        if prompt_type not in self._prompts:
            raise ValueError(f"Тип промпта {prompt_type!r} не найден. Доступные: {list(self._prompts.keys())}")

        df = self._load_dataframe(path, date_col, text_col, title_col)
        df = self._filter_dates(df, date_col, start_date, end_date)
        results = self._process_dataframe(df, date_col, text_col, title_col, prompt_type)
        if output_csv:
            results.to_csv(output_csv, index=False)
            _logger.info("Результаты сохранены в %s", output_csv)
        return results

    # ------------------------------------------------------------------
    # Внутренняя кухня
    # ------------------------------------------------------------------
    def _load_dataframe(
        self, path: str | os.PathLike, date_col: str, text_col: str, title_col: str
    ) -> pd.DataFrame:
        """Читает CSV/TSV, минимальная валидация."""
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
        try:
            df = pd.read_csv(path, sep=sep)
        except FileNotFoundError:
            _logger.error(f"Файл не найден: {path}")
            raise
        except Exception as e:
            _logger.error(f"Ошибка чтения файла {path}: {e}")
            raise

        required_cols = {date_col, text_col, title_col}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise KeyError(f"Файл должен содержать колонки: {missing}")

        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df[text_col] = df[text_col].astype(str).fillna("").str.strip()
        df[title_col] = df[title_col].astype(str).fillna("Unknown").str.strip()
        # Убираем пустые сообщения после strip
        df = df[df[text_col] != ""]
        return df

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        date_col: str,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date).date()]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date).date()]
        return df

    def _process_dataframe(
        self, df: pd.DataFrame, date_col: str, text_col: str, title_col: str, prompt_type: str
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for date, group in df.groupby(date_col):
            messages = list(zip(group[title_col], group[text_col]))
            if not messages:
                _logger.warning("Нет сообщений для даты: %s", date)
                continue
            try:
                result = self._analyze_day(str(date), messages, prompt_type)
                rows.append(result)
            except Exception as e:
                _logger.error("Ошибка анализа дня %s: %s", date, e)
                rows.append({"date": str(date), "summary": f"ERROR: {e}", "impact": {}})

        if not rows:
            _logger.warning("Не было получено ни одного результата анализа.")
            return pd.DataFrame(columns=["date", "summary", "impact"])

        results_df = pd.json_normalize(rows, sep='_')
        results_df.columns = [c.replace('impact_', '') if c.startswith('impact_') else c for c in results_df.columns]

        if 'date' in results_df.columns:
            cols = ['date'] + [col for col in results_df if col != 'date']
            results_df = results_df[cols]
        else:
            _logger.error("Колонка 'date' отсутствует в результатах.")

        return results_df

    # ------------------------------------------------------------------
    # Логика общения с GPT
    # ------------------------------------------------------------------
    def _analyze_day(
        self,
        date: str,
        messages: List[Tuple[str, str]],
        prompt_type: str,
    ) -> Dict[str, Any]:
        """Анализирует один день, при необходимости делит на чанки."""
        chunks = self._split_into_chunks(messages, prompt_type)
        parts = [self._ask_gpt(date, chunk, prompt_type) for chunk in chunks]
        merged = self._merge_parts(date, parts)
        merged["date"] = date
        return merged

    def _split_into_chunks(
        self, messages: List[Tuple[str, str]], prompt_type: str
    ) -> List[List[Tuple[str, str]]]:
        base_prompt = self._prompts[prompt_type].format(
            TICKERS_AND_INDICES=self.tickers_block,
            DATE="YYYY-MM-DD",
            NEWS_LINES="",
        )
        base_tokens = self._count_tokens(base_prompt)
        max_message_tokens = self.max_tokens - base_tokens - 200

        if max_message_tokens <= 0:
            _logger.warning("Max_tokens (%d) слишком мал для базового промпта (%d). Чанкование может быть неэффективным.", self.max_tokens, base_tokens)
            max_message_tokens = 100

        chunks: List[List[Tuple[str, str]]] = []
        current_chunk: List[Tuple[str, str]] = []
        current_tokens = 0

        for title, text in messages:
            message_line = f"{title} : {text}"
            message_tokens = self._count_tokens(message_line) + 1

            if (len(current_chunk) >= self.chunk_limit or
                (current_tokens + message_tokens > max_message_tokens and current_chunk)):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            if message_tokens > max_message_tokens:
                _logger.warning("Сообщение для '%s' слишком длинное (%d токенов), может быть обрезано GPT. Лимит: %d", title, message_tokens, max_message_tokens)

            current_chunk.append((title, text))
            current_tokens += message_tokens

        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) > 1:
            _logger.info("Разделено на %d чанков из-за лимита токенов/сообщений.", len(chunks))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Подсчитывает токены, используя tiktoken, если доступен."""
        if not tiktoken:
            return len(text) // 3
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception as e:
            _logger.warning("Ошибка подсчета токенов tiktoken: %s. Используем примерную оценку.", e)
            return len(text) // 3

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry_error_callback=lambda retry_state: (
            _logger.error("GPT запрос не удался после %d попыток: %s", retry_state.attempt_number, retry_state.outcome.exception()),
            {"summary": f"ERROR: Max retries exceeded - {retry_state.outcome.exception()}", "impact": {}}
        )
    )
    def _chat_completion(self, prompt: str) -> dict:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content: str = response.choices[0].message.content  # type: ignore
            return self._safe_json(content)
        except openai.error.InvalidRequestError as e:
            _logger.error("Ошибка запроса к OpenAI (возможно, превышен лимит токенов): %s", e)
            return {"summary": f"ERROR: Invalid Request - {e}", "impact": {}}
        except Exception as e:
            _logger.error("Неизвестная ошибка при запросе к OpenAI: %s", e)
            raise

    def _ask_gpt(self, date: str, messages: List[Tuple[str, str]], prompt_type: str) -> dict:
        if prompt_type not in self._prompts:
            _logger.error(f"Неизвестный тип промпта: {prompt_type}")
            return {"summary": f"ERROR: Unknown prompt type {prompt_type}", "impact": {}}

        prompt_template = self._prompts[prompt_type]
        formatted_messages = "\n".join(f"{title} : {text}" for title, text in messages)

        try:
            prompt = prompt_template.format(
                TICKERS_AND_INDICES=self.tickers_block,
                DATE=date,
                NEWS_LINES=formatted_messages,
            )
        except KeyError as e:
            _logger.error(f"Ошибка форматирования промпта {prompt_type}: отсутствует ключ {e}")
            return {"summary": f"ERROR: Prompt formatting error - missing key {e}", "impact": {}}

        _logger.debug("Запрос к GPT (prompt_type=%s, date=%s, messages=%d):\n%s",
                      prompt_type, date, len(messages), prompt[:500] + "...")

        result = self._chat_completion(prompt)
        _logger.debug("Ответ от GPT: %s", result)
        return result

    # ------------------------------------------------------------------
    # Обработка JSON
    # ------------------------------------------------------------------
    def _safe_json(self, raw: str) -> dict:
        """Гарантирует валидный JSON, пытаемся поправить мелкие ошибки."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            _logger.warning("JSON ошибка: %s. Попытка исправить", e)
            fixed = self._repair_json(raw)
            try:
                return json.loads(fixed)
            except Exception:
                _logger.error("Не удалось исправить JSON")
                raise

    @staticmethod
    def _repair_json(text: str) -> str:
        """Грубый фикс: отрезаем всё до первой '{' и после последней '}'."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start > end:
            _logger.warning("Не удалось найти '{' и '}' для исправления JSON.")
            return text
        return text[start : end + 1]

    @staticmethod
    def _merge_parts(date: str, parts: List[dict]) -> dict:
        """Объединяет результаты из нескольких чанков для одного дня."""
        if not parts:
            _logger.warning("Нет частей для объединения для даты %s", date)
            return {"summary": "No data processed for this day.", "impact": {}}

        valid_parts = []
        error_summaries = []
        for i, part in enumerate(parts):
            if isinstance(part, dict) and "impact" in part and "summary" in part and not part["summary"].startswith("ERROR:"):
                valid_parts.append(part)
            else:
                error_summary = part.get("summary", f"Unknown error in part {i+1}") if isinstance(part, dict) else f"Invalid data structure in part {i+1}: {part}"
                error_summaries.append(error_summary)
                _logger.error("Ошибка в части %d для даты %s: %s", i+1, date, error_summary)

        if not valid_parts:
            _logger.error("Нет валидных частей для объединения для даты %s. Ошибки: %s", date, "; ".join(error_summaries))
            summary = error_summaries[0] if error_summaries else "All parts failed processing."
            return {"summary": summary, "impact": {}}

        final_summary_prefix = ""
        if error_summaries:
            final_summary_prefix = f"WARNING: {len(error_summaries)} part(s) failed. "
            _logger.warning("Для даты %s были ошибки в %d/%d частях.", date, len(error_summaries), len(parts))

        if len(valid_parts) == 1 and not error_summaries:
            return valid_parts[0]

        merged_impact: Dict[str, List[float]] = {}
        summaries = []
        for part in valid_parts:
            summaries.append(str(part.get("summary", "")))
            impact_data = part.get("impact", {})
            if isinstance(impact_data, dict):
                for ticker, value in impact_data.items():
                    try:
                        num_value = float(value)
                        merged_impact.setdefault(str(ticker), []).append(num_value)
                    except (ValueError, TypeError):
                        _logger.warning("Некорректное значение '%s' для тикера '%s' в части для даты %s. Пропускаем.", value, ticker, date)
            else:
                _logger.warning("Некорректный формат 'impact' в части для даты %s: %s. Пропускаем.", date, impact_data)

        final_impact = {
            ticker: round(sum(values) / len(values), 2)
            for ticker, values in merged_impact.items() if values
        }

        final_summary = final_summary_prefix + " ".join(filter(None, summaries))
        max_summary_len = 1000
        if len(final_summary) > max_summary_len:
            final_summary = final_summary[:max_summary_len-3] + "..."

        return {"summary": final_summary, "impact": final_impact}

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------
    @staticmethod
    def _load_tickers(config_path: str | os.PathLike) -> str:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        companies = cfg.get("companies", {})
        indices = cfg.get("indices", {})
        
        # Создаем список строк для вывода
        ticker_lines = []
        
        # Обрабатываем компании в новом формате
        for ticker, data in companies.items():
            if isinstance(data, dict) and 'names' in data:
                # Добавляем все возможные названия компании
                joined_names = ", ".join(data['names'])
                ticker_lines.append(f"{ticker} : {joined_names}")
            elif isinstance(data, str):
                # Старый формат для обратной совместимости
                ticker_lines.append(f"{ticker} : {data}")
        
        # Индексы оставляем без изменений
        for ticker, description in indices.items():
            ticker_lines.append(f"{ticker} : {description}")
        
        return "\n".join(ticker_lines) 