"""
Amanat-Terra | Модуль 1: Сбор данных со спутников
=================================================
Поддерживает:
  - Sentinel Hub API (production)
  - Демо-режим с синтетическими данными (для презентации жюри)

Использование:
    from data_acquisition import SatelliteDataClient
    client = SatelliteDataClient(use_demo=True)
    data = client.get_vegetation_data(geojson_coords, start_date, end_date)
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("amanat_terra.acquisition")

CLOUD_THRESHOLD = 0.15  # 15% — максимум допустимой облачности


# ─────────────────────────────────────────────
#  Структуры данных
# ─────────────────────────────────────────────

@dataclass
class SatelliteScene:
    """Один спутниковый снимок с нужными слоями."""
    date: str
    band_red: np.ndarray     # B4  — Red
    band_nir: np.ndarray     # B8  — Near-Infrared
    band_blue: np.ndarray    # B2  — Blue (для облаков)
    cloud_fraction: float    # доля пикселей с облаками [0..1]
    is_valid: bool = True

    def __repr__(self):
        status = "✅" if self.is_valid else "☁️ ОБЛАЧНО"
        return f"<Scene {self.date} | облака={self.cloud_fraction:.1%} | {status}>"


@dataclass
class VegetationTimeSeries:
    """Временной ряд чистых снимков по одному участку."""
    plot_id: str
    geojson: dict
    scenes: list = field(default_factory=list)

    def add_scene(self, scene: SatelliteScene):
        if scene.is_valid:
            self.scenes.append(scene)
            logger.info(f"[{self.plot_id}] Добавлен снимок {scene.date}")

    def get_dates(self):
        return [s.date for s in self.scenes]


# ─────────────────────────────────────────────
#  Клиент данных
# ─────────────────────────────────────────────

class SatelliteDataClient:
    """
    Универсальный клиент для получения спутниковых данных.

    Параметры:
        client_id     — Sentinel Hub Client ID
        client_secret — Sentinel Hub Client Secret
        use_demo      — True для демо без API (для хакатона)
    """

    def __init__(self, client_id: str = "", client_secret: str = "", use_demo: bool = False):
        self.use_demo = use_demo
        if not use_demo:
            self._init_sentinel_hub(client_id, client_secret)
        else:
            logger.info("🎭 Режим DEMO активирован — используются синтетические данные")

    # ── Инициализация Sentinel Hub ──────────────────────────────────────────

    def _init_sentinel_hub(self, client_id: str, client_secret: str):
        """Подключение к Sentinel Hub через официальный SDK."""
        try:
            from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType, bbox_to_dimensions
            config = SHConfig()
            config.sh_client_id = client_id
            config.sh_client_secret = client_secret
            if not config.sh_client_id or not config.sh_client_secret:
                raise ValueError("❌ Укажите client_id и client_secret от Sentinel Hub!")
            self.sh_config = config
            self._sentinel_hub = True
            logger.info("✅ Sentinel Hub подключён")
        except ImportError:
            raise ImportError("Установите: pip install sentinelhub")

    # ── Основной метод получения данных ────────────────────────────────────

    def get_vegetation_data(
        self,
        geojson: dict,
        start_date: str,
        end_date: str,
        plot_id: str = "plot_001",
        resolution: int = 10  # метров на пиксель
    ) -> VegetationTimeSeries:
        """
        Загружает временной ряд спутниковых снимков с cloud masking.

        geojson   — {'type': 'Polygon', 'coordinates': [[lon,lat], ...]}
        start_date, end_date — формат 'YYYY-MM-DD'
        """
        logger.info(f"📡 Запрос данных: {plot_id} | {start_date} → {end_date}")

        if self.use_demo:
            return self._get_demo_data(geojson, start_date, end_date, plot_id)
        else:
            return self._get_sentinel_hub_data(geojson, start_date, end_date, plot_id, resolution)

    # ── Sentinel Hub реализация ─────────────────────────────────────────────

    def _get_sentinel_hub_data(self, geojson, start_date, end_date, plot_id, resolution) -> VegetationTimeSeries:
        from sentinelhub import (SentinelHubRequest, DataCollection, BBox,
                                  CRS, MimeType, bbox_to_dimensions)
        import shapely.geometry as sg

        ts = VegetationTimeSeries(plot_id=plot_id, geojson=geojson)
        coords = geojson["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = BBox([min(lons), min(lats), max(lons), max(lats)], crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        # Evalscript — запрашиваем B2, B4, B8 и встроенный CLM (Cloud Layer Mask)
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B02", "B04", "B08", "CLM"], units: "REFLECTANCE"}],
        output: [{id: "default", bands: 4}],
        mosaicking: "ORBIT"
    };
}
function evaluatePixel(sample) {
    return [sample.B04, sample.B08, sample.B02, sample.CLM];
}
"""
        # Генерируем даты каждые 10 дней
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        intervals = []
        while current < end:
            next_d = current + timedelta(days=10)
            intervals.append((current.strftime("%Y-%m-%d"), min(next_d, end).strftime("%Y-%m-%d")))
            current = next_d

        for (s, e) in intervals:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(DataCollection.SENTINEL2_L2A, time_interval=(s, e))],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox, size=size, config=self.sh_config
            )
            data = request.get_data()
            if not data or data[0] is None:
                continue

            img = data[0]  # shape: (H, W, 4)
            band_red  = img[:, :, 0].astype(float) / 10000.0
            band_nir  = img[:, :, 1].astype(float) / 10000.0
            band_blue = img[:, :, 2].astype(float) / 10000.0
            cloud_mask = img[:, :, 3]

            cloud_fraction = np.mean(cloud_mask > 0)
            is_valid = cloud_fraction <= CLOUD_THRESHOLD

            if not is_valid:
                logger.warning(f"☁️  Снимок {s} пропущен — облачность {cloud_fraction:.1%}")

            scene = SatelliteScene(
                date=s,
                band_red=band_red, band_nir=band_nir, band_blue=band_blue,
                cloud_fraction=cloud_fraction, is_valid=is_valid
            )
            ts.add_scene(scene)

        logger.info(f"✅ Загружено {len(ts.scenes)} чистых снимков из {len(intervals)} интервалов")
        return ts

    # ── DEMO реализация (для хакатона без API) ──────────────────────────────

    def _get_demo_data(self, geojson, start_date, end_date, plot_id) -> VegetationTimeSeries:
        """
        Генерирует реалистичные синтетические данные для степей Казахстана.
        Моделирует: весенний рост → летний спад → осеннее восстановление.
        Участок с деградацией показывает замедленное восстановление.
        """
        ts = VegetationTimeSeries(plot_id=plot_id, geojson=geojson)
        rng = np.random.default_rng(seed=hash(plot_id) % 2**31)

        # Параметры деградации — если plot_id содержит "degraded", делаем участок хуже
        is_degraded = "degrad" in plot_id.lower() or "bad" in plot_id.lower()

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        size = (64, 64)  # 64x64 пикселя

        while current <= end:
            # Cloud masking: ~20% снимков будут облачными
            cloud_fraction = rng.uniform(0, 0.35)
            is_valid = cloud_fraction <= CLOUD_THRESHOLD

            # Сезонный профиль NDVI для степи Казахстана
            doy = current.timetuple().tm_yday  # день года
            # Пик вегетации ~160 (начало июня), основание ~30
            savi_base = 0.25 + 0.35 * np.exp(-0.5 * ((doy - 160) / 60) ** 2)

            if is_degraded:
                # Деградированный участок: ниже базовый уровень + медленное восстановление
                year_factor = max(0.5, 1.0 - 0.12 * (current.year - 2023))
                savi_base *= year_factor * 0.75

            # Генерируем пространственно-коррелированный шум
            noise = rng.normal(0, 0.04, size)
            savi_map = np.clip(savi_base + noise, 0.01, 0.85)

            # Обратный расчёт NIR и Red из SAVI (L=0.5)
            L = 0.5
            # SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
            # Упрощённо: Red = 0.08 + noise_red
            red = np.clip(0.08 + rng.normal(0, 0.01, size), 0.01, 0.3)
            # NIR = Red + savi_map * (red + L) / (1 + L - savi_map)
            numer = savi_map * (red + L)
            denom = (1 + L - savi_map)
            nir = np.clip(red + numer / np.where(denom > 0.01, denom, 0.01), 0.01, 0.95)
            blue = np.clip(0.05 + rng.normal(0, 0.005, size), 0.01, 0.2)

            scene = SatelliteScene(
                date=current.strftime("%Y-%m-%d"),
                band_red=red, band_nir=nir, band_blue=blue,
                cloud_fraction=cloud_fraction, is_valid=is_valid
            )
            ts.add_scene(scene)
            current += timedelta(days=16)  # интервал Sentinel-2

        logger.info(f"🎭 DEMO [{plot_id}]: {len(ts.scenes)} снимков | деградация={'Да' if is_degraded else 'Нет'}")
        return ts


# ─────────────────────────────────────────────
#  Быстрый тест
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Пример участка в Акмолинской области
    geojson_example = {
        "type": "Polygon",
        "coordinates": [[
            [70.4, 51.2], [70.5, 51.2], [70.5, 51.3], [70.4, 51.3], [70.4, 51.2]
        ]]
    }

    client = SatelliteDataClient(use_demo=True)

    # Хороший участок
    good = client.get_vegetation_data(geojson_example, "2023-01-01", "2024-12-31", "plot_good")
    print(f"\n✅ Хороший участок: {len(good.scenes)} снимков")

    # Деградированный участок
    bad = client.get_vegetation_data(geojson_example, "2023-01-01", "2024-12-31", "plot_degraded")
    print(f"⚠️  Деградированный: {len(bad.scenes)} снимков")
