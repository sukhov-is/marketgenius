#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-ready скрипт для кластеризации новостей по эмбеддингам.
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import click
import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
import pandas as pd
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class NewsClusterer:
    """
    Класс для кластеризации новостей на основе косинусного расстояния эмбеддингов
    с учетом штрафов за одинаковый канал и временные различия.
    """

    def __init__(
        self,
        same_channels_penalty: float,
        time_penalty_modifier: float,
        time_shift_hours: float,
        linkage: str = "average",
    ) -> None:
        self.same_channels_penalty = same_channels_penalty
        self.time_penalty_modifier = time_penalty_modifier
        self.time_shift_hours = time_shift_hours
        self.linkage = linkage
        self.use_gpu = GPU_AVAILABLE

    @staticmethod
    def _parse_embeddings(series: pd.Series) -> np.ndarray:
        """Преобразует JSON-строки эмбеддингов в numpy.ndarray"""
        return np.vstack(series.apply(json.loads).values)

    def _compute_penalties(
        self, channels: List[str], datetimes: pd.Series
    ) -> np.ndarray:
        """Вычисляет векторизированные штрафы для пар данных"""
        # штраф за одинаковые каналы
        ch_arr = np.array(channels)
        same = (ch_arr[:, None] == ch_arr[None, :]).astype(float)
        same_penalty = 1.0 + (self.same_channels_penalty - 1.0) * same

        # временные штрафы
        if self.time_penalty_modifier <= 1.0:
            time_penalty = np.ones_like(same_penalty)
        else:
            seconds = datetimes.values.astype('datetime64[ns]').astype('int64') / 1e9
            time_diff_h = (
                np.abs(seconds[:, None] - seconds[None, :]) / 3600.0 - self.time_shift_hours
            )
            time_penalty = 1.0 + expit(time_diff_h) * (self.time_penalty_modifier - 1.0)

        # итоговый штраф
        return np.minimum(1.0, same_penalty * time_penalty)

    def cluster_window(self, df_window: pd.DataFrame) -> List[Dict]:
        """
        Кластеризует набор новостей в одном временном окне
        и возвращает список кластеров в формате dict.
        """
        if df_window.empty or len(df_window) < 2:
            return []

        embeddings = self._parse_embeddings(df_window['embedding'])
        if self.use_gpu and cp is not None:
            try:
                emb_gpu = cp.asarray(embeddings)
                norms = cp.linalg.norm(emb_gpu, axis=1)
                cosine_sim = emb_gpu.dot(emb_gpu.T) / (norms[:, None] * norms[None, :])
                base_dist = cp.asnumpy(1 - cosine_sim)
            except Exception:
                base_dist = pairwise_distances(embeddings, metric='cosine', n_jobs=-1)
        else:
            base_dist = pairwise_distances(embeddings, metric='cosine', n_jobs=-1)

        penalties = self._compute_penalties(
            df_window['channel_name'].tolist(), df_window['datetime']
        )
        distances = np.minimum(1.0, base_dist * penalties)

        clustering = AgglomerativeClustering(
            metric='precomputed', linkage=self.linkage
        )
        labels = clustering.fit_predict(distances)

        clusters: List[Dict] = []
        for lbl in np.unique(labels):
            subset = df_window[labels == lbl]
            if subset.empty:
                continue
            # выбираем новость с максимальной длиной текста
            best = subset.loc[subset['news'].str.len().idxmax()]
            clusters.append({
                'cluster_id': f"{best['datetime'].isoformat()}_{lbl}",
                'datetime': best['datetime'].isoformat(),
                'channel_name': best['channel_name'],
                'message_link': best['message_link'],
                'news': best['news'],
            })
        return clusters


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option(
    '--input-path',
    type=click.Path(exists=True, dir_okay=False),
    default='data/raw/news_with_emb.csv',
    help='CSV с новостями и эмбеддингами',
)
@click.option(
    '--config-path',
    type=click.Path(exists=True, dir_okay=False),
    default='configs/clusterer_config.json',
    help='Конфиг JSON с параметрами кластеризации',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False),
    default='data/raw/',
    help='Директория для результатов',
)
@click.option(
    '--window-days',
    type=float,
    default=1.0,
    show_default=True,
    help='Размер окна в днях',
)
@click.option(
    '--window-step',
    type=float,
    default=1.0,
    show_default=True,
    help='Шаг окна в днях',
)
def main(
    input_path: str,
    config_path: str,
    output_dir: str,
    window_days: float,
    window_step: float,
) -> None:
    """
    Основная функция: читает CSV, выполняет скользящее окно, кластеризацию
    и сохраняет результаты в JSON/CSV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    logger.info('Запуск кластеризации новостей')

    try:
        cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
        clusterer = NewsClusterer(
            same_channels_penalty=cfg['distances']['same_channels_penalty'],
            time_penalty_modifier=cfg['distances']['time_penalty_modifier'],
            time_shift_hours=cfg['distances']['time_shift_hours'],
            linkage=cfg.get('clustering', {}).get('linkage', 'average'),
        )

        df_all = pd.read_csv(
            input_path, parse_dates=['datetime'], encoding='utf-8'
        )
        start = df_all['datetime'].min()
        end = df_all['datetime'].max()
        total = []
        buf = pd.DataFrame()
        window_len = timedelta(days=window_days)
        step = timedelta(days=window_step)
        current = start

        for chunk in pd.read_csv(
            input_path,
            parse_dates=['datetime'],
            chunksize=50000,
            encoding='utf-8',
        ):
            chunk = pd.concat([buf, chunk]).sort_values('datetime')
            while current + window_len <= chunk['datetime'].max():
                win = chunk[
                    (chunk['datetime'] >= current) &
                    (chunk['datetime'] < current + window_len)
                ]
                if not win.empty:
                    total.extend(clusterer.cluster_window(win))
                current += step
            buf = chunk[chunk['datetime'] >= current]

        # остаток
        if not buf.empty:
            total.extend(clusterer.cluster_window(buf))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = {
            'total_clusters': len(total),
            'clustering_date': datetime.now().isoformat(),
            'clusters': total,
        }
        json_path = out_dir / 'all_clusters.json'
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f'Сохранен файл: {json_path}')

        if total:
            df_out = pd.DataFrame(total)
            df_out['datetime'] = pd.to_datetime(df_out['datetime'])
            df_out = df_out.sort_values('datetime')
            csv_path = out_dir / 'earliest_news.csv'
            df_out.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f'CSV с ранними новостями: {csv_path}')

        logger.info('Кластеризация завершена успешно')
    except Exception:
        logger.exception('Ошибка во время кластеризации')
        sys.exit(1)


if __name__ == '__main__':
    main()