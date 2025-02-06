from typing import List

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore
from tqdm.auto import tqdm

from util import set_random_seed, gen_batch

# Определяем устройство для вычислений - GPU если доступен, иначе CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Класс для создания эмбеддингов текста с использованием трансформер-моделей"""
    
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        device: str = DEVICE,
        pooling_method: str = "default",
        normalize: bool = True,
        text_prefix: str = "",
    ) -> None:
        """
        Инициализация эмбеддера
        Args:
            model_name: название модели из HuggingFace
            batch_size: размер батча для обработки
            max_length: максимальная длина входного текста
            device: устройство для вычислений ('cuda' или 'cpu')
            pooling_method: метод пулинга ('default', 'mean' или 'cls')
            normalize: нормализовать ли выходные эмбеддинги
            text_prefix: префикс, добавляемый к каждому тексту
        """
        set_random_seed(56154)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.text_prefix = text_prefix

    def __call__(self, texts: List[str]) -> torch.Tensor:
        """
        Создает эмбеддинги для списка текстов
        Args:
            texts: список текстов для эмбеддинга
        Returns:
            torch.Tensor: матрица эмбеддингов размера (len(texts) x hidden_size)
        """
        # Инициализация тензора для хранения эмбеддингов
        embeddings: torch.Tensor = torch.zeros(
            (len(texts), self.model.config.hidden_size)
        )
        
        total = len(texts) // self.batch_size + 1
        desc = "Calc embeddings"
        
        # Добавление префикса к текстам, если задан
        if self.text_prefix:
            texts = [self.text_prefix + text for text in texts]
            
        # Обработка текстов батчами
        for batch_num, batch in enumerate(
            tqdm(gen_batch(texts, self.batch_size), total=total, desc=desc)
        ):
            # Токенизация входных текстов
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            
            with torch.no_grad():
                out = self.model(**inputs)
                
                # Выбор метода пулинга
                if self.pooling_method == "default":
                    # Использование встроенного пулинга модели
                    batch_embeddings = out.pooler_output
                elif self.pooling_method == "mean":
                    # Усреднение по всем токенам с учетом маски внимания
                    hidden_states = out.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    last_hidden = hidden_states.masked_fill(
                        ~attention_mask[..., None].bool(), 0.0
                    )
                    batch_embeddings = (
                        last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                    )
                elif self.pooling_method == "cls":
                    # Использование только [CLS] токена
                    hidden_states = out.last_hidden_state
                    batch_embeddings = hidden_states[:, 0, :]
                    
                # Нормализация эмбеддингов если требуется
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings)
                    
            # Сохранение эмбеддингов батча в общий тензор
            start_index = batch_num * self.batch_size
            end_index = (batch_num + 1) * self.batch_size
            embeddings[start_index:end_index, :] = batch_embeddings
            
        return embeddings
