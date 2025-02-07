import json
import os
from typing import Dict, List, Any

import numpy as np
from scipy.special import expit  # type: ignore
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore

from nyan.clusters import Cluster
from nyan.document import Document


class Clusterer:
    """Класс для кластеризации документов на основе их эмбеддингов и дополнительных параметров"""
    
    def __init__(self, config_path: str):
        """
        Инициализация кластеризатора
        Args:
            config_path: путь к конфигурационному файлу
        """
        assert os.path.exists(config_path)
        with open(config_path) as r:
            self.config: Dict[str, Any] = json.load(r)

    def __call__(self, docs: List[Document]) -> List[Cluster]:
        """
        Выполняет кластеризацию документов
        Args:
            docs: список документов для кластеризации
        Returns:
            список кластеров
        """
        assert docs, "No docs for clusterer"

        # Получение параметров из конфигурации
        distances_config = self.config["distances"]
        same_channels_penalty = distances_config.get("same_channels_penalty", 1.0)  # Штраф для документов из одного канала
        fix_same_channels = same_channels_penalty > 1.0
        time_penalty_modifier = distances_config.get("time_penalty_modifier", 1.0)  # Модификатор штрафа по времени
        fix_time = time_penalty_modifier > 1.0
        time_shift_hours = distances_config.get("time_shift_hours", 4)  # Временной сдвиг в часах
        ntp_issues = distances_config.get("no_time_penalty_issues", tuple())  # Исключения для временного штрафа
        image_bonus = distances_config.get("image_bonus", 0.0)  # Бонус за одинаковые изображения
        fix_images = image_bonus > 0.0
        if fix_images:
            image_idx2cluster = self.find_image_duplicates(docs)

        # Инициализация матрицы эмбеддингов
        min_distance = 0.0
        max_distance = 1.0
        assert docs[0].embedding
        dim = len(docs[0].embedding)
        # Создаем матрицу для хранения эмбеддингов всех документов
        embeddings = np.zeros((len(docs), dim), dtype=np.float32)
        for i, doc in enumerate(docs):
            embeddings[i, :] = doc.embedding

        # Вычисляем попарные косинусные расстояния между документами
        distances = pairwise_distances(
            embeddings, metric="cosine", force_all_finite=False
        )

        # Корректируем расстояния с учетом различных факторов
        for i1, doc1 in enumerate(docs):
            for i2, doc2 in enumerate(docs):
                if i1 == i2:
                    continue

                # Проверяем, применим ли временной штраф к данным документам
                is_time_fixable_issues = (
                    doc1.issue not in ntp_issues or doc2.issue not in ntp_issues
                )

                # Применяем штраф для документов из одного канала
                if fix_same_channels and doc1.channel_id == doc2.channel_id:
                    distances[i1, i2] = min(
                        max_distance, distances[i1, i2] * same_channels_penalty
                    )
                    continue

                # Проверяем наличие одинаковых изображений
                max_images_count = max(
                    len(doc1.embedded_images), len(doc2.embedded_images)
                )
                min_images_count = min(
                    len(doc1.embedded_images), len(doc2.embedded_images)
                )
                # Определяем, находятся ли изображения в одном кластере
                is_same_image_cluster = image_idx2cluster.get(
                    i1, -1
                ) == image_idx2cluster.get(i2, -2)

                # Применяем бонус за одинаковые изображения
                if (
                    fix_images
                    and min_images_count >= 1
                    and max_images_count <= 2
                    and is_same_image_cluster
                ):
                    distances[i1, i2] = max(
                        min_distance, distances[i1, i2] * (1.0 - image_bonus)
                    )

                # Применяем временной штраф
                if fix_time and is_time_fixable_issues:
                    # Вычисляем разницу во времени публикации
                    time_diff = abs(doc1.pub_time - doc2.pub_time)
                    # Вычисляем сдвиг в часах относительно порогового значения
                    hours_shifted = (time_diff / 3600) - time_shift_hours
                    # Применяем сигмоидную функцию для плавного изменения штрафа
                    time_penalty = 1.0 + expit(hours_shifted) * (
                        time_penalty_modifier - 1.0
                    )
                    distances[i1, i2] = min(
                        max_distance, distances[i1, i2] * time_penalty
                    )

        # Выполняем иерархическую кластеризацию
        clustering = AgglomerativeClustering(**self.config["clustering"])
        labels = clustering.fit_predict(distances).tolist()

        # Группируем документы по кластерам
        indices: List[List[int]] = [[] for _ in range(max(labels) + 1)]
        for index, label in enumerate(labels):
            indices[label].append(index)

        # Создаем объекты кластеров и сохраняем информацию о расстояниях
        clusters = []
        for doc_indices in indices:
            cluster = Cluster()
            for index in doc_indices:
                cluster.add(docs[index])
            # Сохраняем матрицу расстояний для документов внутри кластера
            doc_indices_np = np.array(doc_indices)
            cluster.save_distances(distances[doc_indices_np, doc_indices_np])
            clusters.append(cluster)
        return clusters

    def find_image_duplicates(self, docs: List[Document]) -> Dict[int, int]:
        """
        Находит дубликаты изображений в документах
        Args:
            docs: список документов
        Returns:
            словарь соответствия индекса документа и метки кластера изображения
        """
        # Проверяем минимальное количество документов
        if len(docs) < 2:
            return dict()

        # Сбор эмбеддингов изображений и их привязка к документам
        embeddings, image2doc = [], []
        for i, doc in enumerate(docs):
            for image in doc.embedded_images:
                embeddings.append(image["embedding"])
                image2doc.append(i)
        
        # Проверяем наличие изображений
        if not image2doc:
            return dict()
        if len(embeddings) < 2:
            return dict()

        # Создаем матрицу эмбеддингов изображений
        dim = len(embeddings[0])
        np_embeddings = np.zeros((len(image2doc), dim), dtype=np.float32)
        for i, embedding in enumerate(embeddings):
            np_embeddings[i, :] = embedding

        # Настраиваем и выполняем кластеризацию изображений
        clustering = AgglomerativeClustering(
            n_clusters=None,  # Автоматическое определение количества кластеров
            affinity="cosine",  # Используем косинусное расстояние
            linkage="average",  # Используем среднее расстояние между кластерами
            distance_threshold=0.02,  # Порог расстояния для объединения кластеров
        )

        # Выполняем кластеризацию и создаем маппинг документ -> кластер изображения
        labels = clustering.fit_predict(np_embeddings).tolist()
        return {image2doc[i]: l for i, l in enumerate(labels)}
