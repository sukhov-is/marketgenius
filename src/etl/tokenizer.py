from typing import Any

from natasha import (  # type: ignore
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc,
)

class Tokenizer:
    def __init__(self) -> None:
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()        # Для разделения на предложения
        self.morph_vocab = MorphVocab()     # Морфологический словарь
        self.emb = NewsEmbedding()          # Эмбеддинги для морфологического анализа
        self.morph_tagger = NewsMorphTagger(self.emb)  # Морфологический анализатор

    def __call__(self, text: str) -> Any:
        doc = Doc(text)                    # Создание документа
        doc.segment(self.segmenter)        # Сегментация на предложения
        doc.tag_morph(self.morph_tagger)   # Морфологическая разметка
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)  # Лемматизация каждого токена
        return doc.tokens
