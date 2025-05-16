from __future__ import annotations

import json
import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai

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
        all_config_path: str | os.PathLike = "configs/all_companies_config.json",
        chunk_limit: int = 50,  # максимум сообщений в одном запросе
        retries: int = 6,
    ) -> None:
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY не задан")

        self.model = model
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

        # --- Изменения: Загрузка двух конфигов --- 
        try:
            # Загружаем базовый конфиг
            base_cfg = self._load_config_json(config_path)
            self.tickers_block, self._base_tickers_set = self._process_base_config(base_cfg)

            # Загружаем полный конфиг для поиска
            all_cfg = self._load_config_json(all_config_path)
            self._all_tickers_data, self._all_tickers_descriptions = self._process_all_config(all_cfg)

        except FileNotFoundError as e:
             _logger.error(f"Ошибка загрузки конфигурационного файла: {e}")
             raise
        except Exception as e:
             _logger.error(f"Ошибка обработки конфигурационного файла: {e}")
             raise
        # --- Конец изменений --- 

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
        chunks: List[List[Tuple[str, str]]] = []
        current_chunk: List[Tuple[str, str]] = []

        for title, text in messages:
            if len(current_chunk) >= self.chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []

            current_chunk.append((title, text))

        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) > 1:
            _logger.info("Разделено на %d чанков из-за лимита сообщений (%d).", len(chunks), self.chunk_limit)

        return chunks

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry_error_callback=lambda retry_state: (
            _logger.error("GPT запрос не удался после %d попыток: %s", retry_state.attempt_number, retry_state.outcome.exception()),
            {"summary": f"ERROR: Max retries exceeded - {retry_state.outcome.exception()}", "impact": {}}
        )
    )
    def _chat_completion(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            if not hasattr(openai, 'chat') or not hasattr(openai.chat, 'completions'):
                 _logger.warning("OpenAI client might not be initialized in the modern way. Proceeding with openai.ChatCompletion.create.")

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
            )
            content: str = response.choices[0].message.content  # type: ignore
            return self._safe_json(content)
        except openai.error.InvalidRequestError as e:
            _logger.error("Ошибка запроса к OpenAI (InvalidRequestError): %s", e)
            if "response_format" in str(e).lower() and "is not supported with this model" in str(e).lower():
                 _logger.error(f"Модель {self.model} не поддерживает response_format={{'type': 'json_object'}}."
                               " Пожалуйста, используйте совместимую модель (например, gpt-3.5-turbo-1106, gpt-4-1106-preview или новее).")
                 return {"summary": f"ERROR: Model {self.model} does not support JSON mode. - {e}", "impact": {}}
            return {"summary": f"ERROR: Invalid Request - {e}", "impact": {}}
        except Exception as e:
            _logger.error("Неизвестная ошибка при запросе к OpenAI: %s", e, exc_info=True)
            raise

    def _ask_gpt(self, date: str, messages: List[Tuple[str, str]], prompt_type: str) -> dict:
        if prompt_type not in self._prompts:
            _logger.error(f"Неизвестный тип промпта: {prompt_type}")
            return {"summary": f"ERROR: Unknown prompt type {prompt_type}", "impact": {}}

        prompt_template = self._prompts[prompt_type]
        formatted_messages = "\\n".join(f"{title.replace(chr(10), ' ').replace(chr(13), '')} : {text.replace(chr(10), ' ').replace(chr(13), '')}" for title, text in messages)

        system_marker = "####################  SYSTEM  ####################"
        user_marker = "#####################  USER  #####################"
        end_marker = "##################################################"

        try:
            system_start_idx = prompt_template.index(system_marker) + len(system_marker)
            system_end_idx = prompt_template.index(end_marker, system_start_idx)
            system_template_part = prompt_template[system_start_idx:system_end_idx].strip()

            user_start_idx = prompt_template.index(user_marker) + len(user_marker)
            user_end_idx = prompt_template.index(end_marker, user_start_idx)
            user_template_part = prompt_template[user_start_idx:user_end_idx].strip()

            # --- Новая логика: Поиск доп. тикеров и формирование блока --- 
            additional_tickers = self._find_additional_tickers(messages)
            current_tickers_block = self.tickers_block
            if additional_tickers:
                additional_lines = [ 
                    self._all_tickers_descriptions.get(ticker, f"{ticker} : Описание не найдено") 
                    for ticker in additional_tickers
                ]
                current_tickers_block = self.tickers_block + "\n" + "\n".join(additional_lines)
                _logger.info(f"Для даты {date} добавлены тикеры: {additional_tickers}")
            # --- Конец новой логики ---

            formatted_system_prompt = system_template_part.format(
                TICKERS_AND_INDICES=current_tickers_block # Используем обновленный блок
                # Добавляем "пустышки" для других возможных ключей
                # DATE="",
                # NEWS_LINES=""
            )

            formatted_user_prompt = user_template_part.format(
                DATE=date,
                NEWS_LINES=formatted_messages
            )
        except (ValueError, KeyError, IndexError) as e:
            _logger.error(f"Ошибка разбора или форматирования шаблона промпта '{prompt_type}' для даты {date}: {e}. "
                          f"Проверьте маркеры '{system_marker}', '{user_marker}', '{end_marker}' и плейсхолдеры.")
            return {"summary": f"ERROR: Prompt template processing error - {e}", "impact": {}}

        _logger.debug("Запрос к GPT (prompt_type=%s, date=%s, messages=%d):\\nSYSTEM: %s...\\nUSER: %s...",
                      prompt_type, date, len(messages), formatted_system_prompt[:200], formatted_user_prompt[:300])

        result = self._chat_completion(formatted_system_prompt, formatted_user_prompt)
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

    # --- Новые методы для загрузки и обработки конфигов --- 
    @staticmethod
    def _load_config_json(path: str | os.PathLike) -> dict:
        """Загружает JSON конфиг из файла."""
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            _logger.error(f"Конфигурационный файл не найден: {path}")
            raise
        except json.JSONDecodeError as e:
            _logger.error(f"Ошибка парсинга JSON в файле {path}: {e}")
            raise

    @staticmethod
    def _process_base_config(cfg: dict) -> Tuple[str, set]:
        """Обрабатывает базовый конфиг, возвращает блок тикеров для промпта и множество базовых тикеров."""
        ticker_lines = []
        base_tickers = set()
        companies = cfg.get("companies", {})
        indices = cfg.get("indices", {})

        for ticker, data in companies.items():
            base_tickers.add(ticker)
            if isinstance(data, dict) and 'names' in data:
                joined_names = ", ".join(data['names'])
                ticker_lines.append(f"{ticker} : {joined_names}")
            elif isinstance(data, str):
                ticker_lines.append(f"{ticker} : {data}")

        for ticker, description in indices.items():
            base_tickers.add(ticker)
            ticker_lines.append(f"{ticker} : {description}")

        return "\n".join(ticker_lines), base_tickers

    @staticmethod
    def _process_all_config(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Обрабатывает полный конфиг, создавая структуры для поиска.
        Возвращает:
         - all_tickers_data: Словарь {lower_name_or_ticker -> main_ticker}
         - all_tickers_descriptions: Словарь {main_ticker -> description_line}
        """
        all_tickers_data = {}
        all_tickers_descriptions = {}
        companies = cfg.get("companies", {})
        indices = cfg.get("indices", {})

        for ticker, data in companies.items():
            ticker_lower = ticker.lower()
            description_line = ""
            names_to_add = {ticker_lower}

            if isinstance(data, dict) and 'names' in data:
                joined_names = ", ".join(data['names'])
                description_line = f"{ticker} : {joined_names}"
                for name in data['names']:
                    names_to_add.add(name.lower())
            elif isinstance(data, str):
                description_line = f"{ticker} : {data}"
                names_to_add.add(data.lower())
            
            if description_line:
                 all_tickers_descriptions[ticker] = description_line
                 for name in names_to_add:
                      # Не перезаписываем, если имя уже занято (на случай дубликатов в конфиге)
                      if name not in all_tickers_data:
                          all_tickers_data[name] = ticker

        for ticker, description in indices.items():
            ticker_lower = ticker.lower()
            description_line = f"{ticker} : {description}"
            all_tickers_descriptions[ticker] = description_line
            # Добавляем и тикер, и описание (если оно короткое) для поиска
            if ticker_lower not in all_tickers_data:
                all_tickers_data[ticker_lower] = ticker
            # Опционально: можно добавить и description.lower() в all_tickers_data, 
            # но это может дать ложные срабатывания. Пока не будем.

        return all_tickers_data, all_tickers_descriptions

    def _find_additional_tickers(self, messages: List[Tuple[str, str]]) -> set:
        """Ищет упоминания тикеров/названий из полного конфига в сообщениях как отдельные слова/фразы,
           возвращает множество основных тикеров, не входящих в базовый набор.
        """
        found_tickers = set()
        # Объединяем заголовки и текст для поиска, приводим к нижнему регистру
        full_text = " \n ".join(f"{title} {text}" for title, text in messages).lower()

        # Используем поиск по границам слов с помощью регулярных выражений.
        # (?<!\w) гарантирует, что перед ключом нет буквенно-цифрового символа.
        # (?!\w) гарантирует, что после ключа нет буквенно-цифрового символа.
        # Это позволяет находить ключ как "целое слово" или фразу.
        for name_or_ticker_key, main_ticker in self._all_tickers_data.items():
            # name_or_ticker_key уже в нижнем регистре (из _process_all_config)
            # Экранируем ключ на случай, если он содержит спецсимволы для regex
            escaped_key = re.escape(name_or_ticker_key)
            # Формируем паттерн для поиска. re.IGNORECASE не нужен, т.к. full_text и key уже в lower case.
            pattern = rf"(?<!\w){escaped_key}(?!\w)"
            if re.search(pattern, full_text):
                found_tickers.add(main_ticker)

        # Вычисляем тикеры, которые были найдены дополнительно к базовому набору
        additional_tickers = found_tickers - self._base_tickers_set
        if additional_tickers:
            _logger.debug(f"Найдены дополнительные тикеры для чанка: {additional_tickers}")

        return additional_tickers

    # --- Убираем старый статический метод _load_tickers --- 
    # @staticmethod
    # def _load_tickers(config_path: str | os.PathLike) -> str:
    #     cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    #     ...
    #     return "\n".join(ticker_lines) 