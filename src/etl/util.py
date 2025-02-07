import os
import json
import random
from typing import TypeVar, List, Any, Iterable, Dict, Type
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch


def read_jsonl(file_path: str, sample_rate: float = 1.0) -> Iterable[Dict[str, Any]]:
    """
    Читает JSONL файл построчно с возможностью выборки записей.
    
    Args:
        file_path: Путь к JSONL файлу
        sample_rate: Доля записей для чтения (от 0 до 1)
    
    Yields:
        Словарь с данными из каждой строки JSONL
    """
    assert os.path.exists(file_path)
    with open(file_path) as r:
        for line in r:
            if not line:
                continue
            if random.random() > sample_rate:
                continue
            yield json.loads(line)


def write_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    """
    Записывает список словарей в JSONL файл.
    
    Args:
        file_path: Путь к выходному файлу
        records: Список словарей для записи
    """
    with open(file_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def get_current_ts() -> int:
    """
    Возвращает текущее время в формате UNIX timestamp (UTC).
    
    Returns:
        Текущее время в секундах с начала эпохи
    """
    return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())


def ts_to_dt(timestamp: int, offset: int = 3) -> datetime:
    """
    Конвертирует UNIX timestamp в объект datetime с учетом временной зоны.
    
    Args:
        timestamp: Время в секундах с начала эпохи
        offset: Смещение временной зоны в часах (по умолчанию UTC+3)
    
    Returns:
        Объект datetime с указанной временной зоной
    """
    return datetime.fromtimestamp(timestamp, timezone(timedelta(hours=offset)))


T = TypeVar("T", bound="Serializable")


@dataclass
class Serializable:
    """
    Базовый класс для объектов, которые можно сериализовать в JSON и десериализовать из него.
    
    Предоставляет методы для:
    - Создания объекта из словаря
    - Конвертации объекта в словарь
    - Сериализации в JSON строку
    - Десериализации из JSON строки
    """
    @classmethod
    def fromdict(cls: Type[T], d: Dict[str, Any]) -> T:
        if d is None:
            return None
        keys = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in keys}
        return cls(**d)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def deserialize(cls: Type[T], line: str) -> T:
        return cls.fromdict(json.loads(line))

    def serialize(self) -> str:
        return json.dumps(self.asdict(), ensure_ascii=False)


def set_random_seed(seed: int) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел для воспроизводимости результатов.
    
    Затрагивает:
    - Python random
    - NumPy
    - PyTorch (CPU и CUDA)
    - CUDA cuBLAS
    - Python hash seed
    
    Args:
        seed: Значение для инициализации генераторов
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def gen_batch(records: List[Any], batch_size: int) -> Iterable[List[Any]]:
    """
    Разбивает список на батчи заданного размера.
    
    Args:
        records: Исходный список элементов
        batch_size: Размер каждого батча
    
    Yields:
        Список элементов размера batch_size (последний может быть меньше)
    """
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start:batch_end]
        batch_start = batch_end
        yield batch
