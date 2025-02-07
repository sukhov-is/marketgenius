import json
import os
import traceback
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, TypeVar, Sequence
from typing import Counter as CounterT
from collections import Counter, defaultdict
from functools import cached_property

from jinja2 import Template

from nyan.client import MessageId
from nyan.document import Document
from nyan.mongo import get_clusters_collection
from nyan.title import choose_title
from nyan.openai import openai_completion


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
T = TypeVar("T")


class Cluster:
    """Класс для представления кластера документов с похожим содержанием"""
    
    def __init__(self) -> None:
        """Инициализация кластера"""
        # Список документов в кластере
        self.docs: List[Document] = list()
        # Словарь для быстрого доступа к документам по URL
        self.url2doc: Dict[str, Document] = dict()
        # Уникальный идентификатор кластера
        self.clid: Optional[int] = None
        # Флаг важности кластера
        self.is_important: bool = False

        # Время создания кластера
        self.create_time: Optional[int] = None
        # Список идентификаторов сообщений, связанных с кластером
        self.messages: List[MessageId] = list()

        # Матрица расстояний между документами в кластере
        self.distances: Optional[List[float]] = None

        # Кэшированные значения для оптимизации
        self.saved_annotation_doc: Optional[Document] = None  # Документ с аннотацией
        self.saved_first_doc: Optional[Document] = None      # Первый документ
        self.saved_hash: Optional[str] = None               # Хеш кластера
        self.saved_diff: Optional[List[Dict[str, Any]]] = None  # Различия между документами

    def add(self, doc: Document) -> None:
        """
        Добавление документа в кластер
        Args:
            doc: документ для добавления
        """
        self.docs.append(doc)
        self.url2doc[doc.url] = doc

    def save_distances(self, distances: List[float]) -> None:
        self.distances = distances

    def has(self, doc: Document) -> bool:
        return doc.url in self.url2doc

    def changed(self) -> bool:
        return self.hash != self.saved_hash

    @property
    def pub_time(self) -> int:
        """Время публикации первого документа в кластере"""
        return self.first_doc.pub_time

    @cached_property
    def fetch_time(self) -> int:
        """
        Максимальное время получения документов в кластере
        Returns:
            время в unix timestamp
        """
        times = [doc.fetch_time for doc in self.docs if doc.fetch_time]
        if not times:
            return 0
        return max(times)

    @property
    def views(self) -> int:
        """Общее количество просмотров всех документов"""
        return sum([doc.views for doc in self.docs])

    @property
    def debiased_views(self) -> int:
        """
        Скорректированное количество просмотров с учетом выбросов
        Если документ имеет намного больше просмотров чем остальные,
        его значение сглаживается до уровня второго по популярности
        """
        views = [doc.views for doc in self.unique_docs]
        if len(views) <= 2:
            return sum(views)
        views.sort(reverse=True)
        views[0] = views[1]  # Сглаживание выброса
        return sum(views)

    @property
    def age(self) -> int:
        return self.fetch_time - self.pub_time_percentile

    @property
    def views_per_hour(self) -> int:
        return int(self.debiased_views / (self.age / 3600))

    @property
    def embedding(self) -> Optional[List[float]]:
        if not self.annotation_doc:
            return None
        return self.annotation_doc.embedding

    @cached_property
    def pub_time_percentile(self) -> int:
        """
        Вычисляет 20-й процентиль времени публикации документов
        для определения примерного начала события
        """
        timestamps = sorted([d.pub_time for d in self.docs])
        return timestamps[len(timestamps) // 5]

    @cached_property
    def images(self) -> Sequence[str]:
        """
        Получает список URL изображений, если они присутствуют
        в достаточном количестве документов кластера
        """
        # Подсчет документов с изображениями
        image_doc_count = sum([bool(doc.images) for doc in self.unique_docs])
        doc_count = len(self.unique_docs)
        if doc_count == 0:
            return tuple()
            
        # Получение URL изображений из аннотации
        images = [
            i["url"] for i in self.annotation_doc.embedded_images if self.annotation_doc
        ]
        if not images:
            return tuple()
            
        # Возвращаем изображения только если они есть в достаточном количестве документов
        if image_doc_count / doc_count >= 0.4 or image_doc_count >= 3:
            return images
        return tuple()

    @cached_property
    def videos(self) -> Sequence[str]:
        videos = self.annotation_doc.videos
        if videos:
            return videos
        return tuple()

    @cached_property
    def cropped_title(self, max_words: int = 14) -> str:
        text = self.annotation_doc.patched_text
        if not text:
            return ""
        words = text.split()
        if len(words) < max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + "..."

    @property
    def urls(self) -> List[str]:
        return list(self.url2doc.keys())

    @property
    def channels(self) -> List[str]:
        return list({d.channel_id for d in self.docs})

    @property
    def first_doc(self) -> Document:
        if self.saved_first_doc:
            return self.saved_first_doc
        return min(self.docs, key=lambda x: x.pub_time)

    @property
    def diff(self) -> List[Dict[str, Any]]:
        """
        Генерирует список различий между документами в кластере
        используя GPT-4 для анализа
        """
        if self.saved_diff is not None:
            return self.saved_diff

        # Загрузка шаблона промпта
        prompt_path: Path = BASE_DIR / "prompts/diff.txt"
        with open(prompt_path) as f:
            template = Template(f.read())
        
        # Генерация промпта с данными документов
        prompt = template.render(docs=self.docs, annotation_doc=self.annotation_doc)
        messages = [{"role": "user", "content": prompt}]

        differences: List[Dict[str, Any]] = []
        try:
            # Получение и парсинг ответа от GPT-4
            content = openai_completion(messages=messages, model_name="gpt-4o")
            content = content[content.find("{") : content.rfind("}") + 1]
            parsed_content: Dict[str, List[Dict[str, Any]]] = json.loads(content)
            differences = parsed_content["differences"]

            # Форматирование ссылок на каналы
            channel_titles = {doc.channel_id: doc.channel_title for doc in self.docs}
            doc_urls = {doc.channel_id: doc.url for doc in self.docs}
            for diff in differences:
                ids = diff["channel_ids"][:3]
                ids = [i for i in ids if i in channel_titles and i in doc_urls]
                channels = [
                    '<a href="{}">{}</a>'.format(doc_urls[i], channel_titles[i])
                    for i in ids
                ]
                diff["channels"] = ", ".join(channels)
        except Exception:
            traceback.print_exc()
            differences = []
        return differences

    @property
    def annotation_doc(self) -> Document:
        if self.saved_annotation_doc is not None:
            return self.saved_annotation_doc
        assert self.docs
        self.saved_annotation_doc = choose_title(self.docs, self.issues)
        return self.saved_annotation_doc

    @cached_property
    def hash(self) -> str:  # noqa: A003
        data = " ".join(sorted({d.channel_id for d in self.docs}))
        data += " " + str(self.views // 100000)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @property
    def unique_docs(self) -> List[Document]:
        return [doc for doc in self.docs if not doc.forward_from]

    @property
    def external_links(self) -> CounterT[str]:
        links: CounterT[str] = Counter()
        used_channels = set()
        for doc in self.unique_docs:
            doc_links = set(doc.links)
            if doc.channel_id in used_channels:
                continue
            for link in doc_links:
                if "t.me" not in link and "http" in link:
                    links[link] += 1
                    used_channels.add(doc.channel_id)
        return links

    @property
    def group(self) -> str:
        """
        Определяет группу кластера на основе распределения документов
        по группам "blue", "red" и "purple"
        Returns:
            строка с названием группы
        """
        groups = [doc.groups["main"] for doc in self.docs if doc.groups]
        if not groups:
            return "purple"

        groups_count = Counter(groups)

        all_count = len(groups)
        blue_part = groups_count["blue"] / all_count
        red_part = groups_count["red"] / all_count

        # Определяем группу на основе преобладания документов
        if blue_part == 0.0 and red_part > 0.5:
            return "red"
        if red_part == 0.0 and blue_part > 0.5:
            return "blue"
        return "purple"

    @property
    def issues(self) -> List[str]:
        """
        Определяет список тем (issues) кластера
        Returns:
            список тем
        """
        # Если есть привязанные сообщения, берем темы из них
        if self.messages:
            return [m.issue for m in self.messages]

        def get_most_common(items: List[T]) -> List[T]:
            """Вспомогательная функция для получения наиболее частых элементов"""
            counter = Counter(items)
            max_count = counter.most_common(1)[0][1]
            return [item for item, count in counter.items() if count == max_count]

        # Получаем наиболее частые темы и категории из документов
        issues: List[str] = get_most_common(
            [doc.issue for doc in self.docs if doc.issue]
        )
        categories: List[str] = get_most_common(
            [doc.category for doc in self.docs if doc.category]
        )

        # Формируем итоговый список тем
        final_issues: List[str] = ["main"]
        final_issues.extend(issues)
        final_issues.extend(categories)
        return list(set(final_issues))

    def get_issue_message(self, issue: str) -> Optional[MessageId]:
        messages = [m for m in self.messages if m.issue == issue]
        if messages:
            return messages[0]
        return None

    def get_url(self, host: str, issue: str) -> Optional[str]:
        message = self.get_issue_message(issue)
        if not message:
            return None
        message_id = message.message_id
        return f"{host}/{message_id}"

    def asdict(self) -> Dict[str, Any]:
        """
        Сериализует кластер в словарь
        Returns:
            словарь с данными кластера
        """
        docs = [d.asdict(is_short=True) for d in self.docs]
        annotation_doc = self.annotation_doc.asdict()
        first_doc = self.first_doc.asdict(is_short=True)
        return {
            "clid": self.clid,
            "docs": docs,
            "messages": [m.asdict() for m in self.messages],
            "annotation_doc": annotation_doc,
            "first_doc": first_doc,
            "hash": self.hash,
            "diff": self.diff,
            "is_important": self.is_important,
            "create_time": self.create_time,
        }

    @classmethod
    def fromdict(cls, d: Dict[str, Any]) -> "Cluster":
        cluster = cls()
        cluster.clid = d.get("clid")

        for doc in d["docs"]:
            cluster.add(Document.fromdict(doc))

        if "message" in d:
            cluster.messages = [MessageId.fromdict(d["message"])]
        elif "messages" in d:
            cluster.messages = [MessageId.fromdict(m) for m in d["messages"]]
        elif "message_id" in d:
            cluster.messages = [MessageId(message_id=d["message_id"])]

        annotation_doc_dict = d.get("annotation_doc")
        if annotation_doc_dict:
            cluster.saved_annotation_doc = Document.fromdict(annotation_doc_dict)
        first_doc_dict = d.get("first_doc")
        if first_doc_dict:
            cluster.saved_first_doc = Document.fromdict(first_doc_dict)
        cluster.saved_hash = d.get("hash")
        cluster.saved_diff = d.get("diff", None)
        cluster.is_important = d.get("is_important", False)
        cluster.create_time = d.get("create_time", None)

        return cluster

    def serialize(self) -> str:
        return json.dumps(self.asdict(), ensure_ascii=False)

    @classmethod
    def deserialize(cls, line: str) -> "Cluster":
        return cls.fromdict(json.loads(line))


class Clusters:
    """Класс для управления коллекцией кластеров"""

    def __init__(self) -> None:
        """Инициализация коллекции кластеров"""
        # Словарь для быстрого доступа к кластерам по их ID
        self.clid2cluster: Dict[int, Cluster] = dict()
        # Словарь для быстрого доступа к кластерам по ID сообщений
        self.message2cluster: Dict[MessageId, Cluster] = dict()
        # Максимальный ID кластера
        self.max_clid: int = 60000

    def find_similar(
        self,
        cluster: Cluster,
        issue_name: str,
        min_size_ratio: float = 0.25,
        min_intersection_ratio: float = 0.25,
    ) -> Optional[Cluster]:
        """
        Поиск похожего кластера среди существующих
        Args:
            cluster: кластер для поиска похожего
            issue_name: название темы
            min_size_ratio: минимальное отношение размеров кластеров
            min_intersection_ratio: минимальное отношение пересечения
        Returns:
            похожий кластер или None
        """
        # Собираем сообщения для URL'ов из кластера
        messages = list()
        for url in cluster.urls:
            message = self.urls2messages[issue_name].get(url)
            if message is None:
                continue
            messages.append(message)
        if not messages:
            return None

        # Находим наиболее частое сообщение
        message, intersection_count = Counter(messages).most_common()[0]
        old_cluster = self.message2cluster.get(message)
        if old_cluster is None:
            return None

        # Вычисляем метрики схожести кластеров
        new_cluster_size = len(cluster.urls)
        old_cluster_size = len(old_cluster.urls)
        intersection_ratio = intersection_count / new_cluster_size
        intersection_ratio = min(
            intersection_ratio, intersection_count / old_cluster_size
        )
        size_ratio = new_cluster_size / old_cluster_size

        # Проверяем условия схожести
        if size_ratio < min_size_ratio or intersection_ratio < min_intersection_ratio:
            return None
        return old_cluster

    def get_embedded_clusters(self, current_ts: int, issue: str) -> List[Cluster]:
        """
        Получает список кластеров с эмбеддингами для заданной темы
        Args:
            current_ts: текущее время
            issue: тема
        Returns:
            список кластеров
        """
        filtered_clusters = []
        for cluster in self.clid2cluster.values():
            # Фильтруем кластеры по наличию эмбеддинга, сообщений,
            # временному окну и теме
            if not cluster.embedding:
                continue
            if not cluster.messages:
                continue
            if abs(cluster.pub_time - current_ts) > 24 * 3600:
                continue
            if issue not in cluster.issues:
                continue
            filtered_clusters.append(cluster)
        return filtered_clusters

    def add(self, cluster: Cluster) -> None:
        if cluster.clid is None:
            self.max_clid += 1
            cluster.clid = self.max_clid

        for message in cluster.messages:
            self.message2cluster[message] = cluster
        self.clid2cluster[cluster.clid] = cluster
        self.max_clid = max(self.max_clid, cluster.clid)

    def __len__(self) -> int:
        return len(self.clid2cluster)

    @cached_property
    def urls2messages(self) -> Dict[str, Dict[str, MessageId]]:
        result: Dict[str, Dict[str, MessageId]] = defaultdict(dict)
        for _, cluster in self.clid2cluster.items():
            for url in cluster.urls:
                for message in cluster.messages:
                    result[message.issue][url] = message
        return result

    def update_documents(self, documents: List[Document]) -> int:
        url2doc = {doc.url: doc for doc in documents}
        updates_count = 0
        for _, cluster in self.clid2cluster.items():
            for doc_index, doc in enumerate(cluster.docs):
                url = doc.url
                if url not in url2doc:
                    continue
                new_doc = url2doc[url]
                if (
                    doc.patched_text == new_doc.patched_text
                    and doc.views == new_doc.views
                ):
                    continue
                cluster.docs[doc_index] = new_doc
                cluster.url2doc[url] = new_doc
                if (
                    cluster.saved_annotation_doc
                    and cluster.saved_annotation_doc.url == url
                ):
                    cluster.saved_annotation_doc = new_doc
                updates_count += 1
        return updates_count

    def save(self, path: str) -> None:
        temp_path = path + ".new"
        with open(path + ".new", "w") as w:
            for _, cluster in sorted(self.clid2cluster.items()):
                w.write(cluster.serialize() + "\n")
        shutil.move(temp_path, path)

    @classmethod
    def load(cls, path: str) -> "Clusters":
        assert os.path.exists(path)
        clusters = cls()
        with open(path) as r:
            for line in r:
                clusters.add(Cluster.deserialize(line))
        return clusters

    def save_to_mongo(self, mongo_config_path: str, only_new: bool = True) -> int:
        collection = get_clusters_collection(mongo_config_path)
        if not self.clid2cluster:
            return 0
        max_cluster_fetch_time = max(
            [cl.fetch_time for cl in self.clid2cluster.values()]
        )
        saved_count = 0
        for clid, cluster in sorted(self.clid2cluster.items()):
            if only_new and max_cluster_fetch_time - cluster.fetch_time > 24 * 3600:
                continue
            saved_count += 1
            collection.replace_one({"clid": clid}, cluster.asdict(), upsert=True)
        return saved_count

    @classmethod
    def load_from_mongo(
        cls, mongo_config_path: str, current_ts: int, offset: int
    ) -> "Clusters":
        collection = get_clusters_collection(mongo_config_path)
        clusters_dicts = list(
            collection.find({"create_time": {"$gte": current_ts - offset}})
        )
        clusters = cls()
        for cluster_dict in clusters_dicts:
            clusters.add(Cluster.fromdict(cluster_dict))
        return clusters
