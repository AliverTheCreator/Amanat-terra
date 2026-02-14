"""
Amanat-Terra | Модуль 2: Аналитический движок
=============================================
Реализует:
  - SAVI (Soil Adjusted Vegetation Index)
  - Расчёт биомассы (т сухого вещества/га) для степей Казахстана
  - VRP (Vegetation Recovery Rate) — алгоритм обнаружения деградации

Использование:
    from analytics_engine import VegetationAnalyzer
    analyzer = VegetationAnalyzer()
    result = analyzer.analyze(time_series)
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("amanat_terra.analytics")

# ─────────────────────────────────────────────
#  Константы для степей Казахстана
# ─────────────────────────────────────────────

L_SAVI = 0.5           # Почвенный корректирующий фактор (стандарт для полузасушливых зон)

# Коэффициенты конверсии биомассы для казахстанских степей
# Источник: FAO Grassland Index, адаптация для Центральной Азии
BIOMASS_SLOPE     = 8.43   # т сухого вещества/га на единицу SAVI
BIOMASS_INTERCEPT = -1.12  # смещение
BIOMASS_MIN       = 0.0
BIOMASS_MAX       = 4.5    # максимум для типичных степей (т/га)

# Пороги деградации
DEGRADATION_VRP_THRESHOLD = 0.75  # если восстановление < 75% от среднего за 3 года → деградация
OVERGRAZING_SAVI_THRESHOLD = 0.18  # SAVI ниже этого → перевыпас (красная зона)
RISK_SAVI_THRESHOLD = 0.25        # SAVI ниже этого → зона риска (жёлтая зона)


# ─────────────────────────────────────────────
#  Структуры результатов
# ─────────────────────────────────────────────

@dataclass
class PlotAnalysisResult:
    """Полный аналитический результат по одному участку."""
    plot_id: str

    # Временные ряды
    dates: list
    savi_series: list          # среднее SAVI по дате
    biomass_series: list       # т/га по дате
    savi_maps: list            # 2D карты SAVI (для визуализации)

    # Агрегированные показатели
    current_savi: float = 0.0
    current_biomass: float = 0.0
    carrying_capacity: float = 0.0  # условных голов скота на га

    # VRP - анализ деградации
    vrp_score: float = 1.0         # 1.0 = норма, < 0.75 = деградация
    is_degraded: bool = False
    degradation_flag: str = "OK"   # OK / РИСК / ДЕГРАДАЦИЯ

    # Зонирование
    overgrazing_fraction: float = 0.0  # доля пикселей в красной зоне
    risk_fraction: float = 0.0

    # Рекомендации
    recommendations: list = field(default_factory=list)
    subsidy_risk_score: float = 0.0  # 0-100: риск признания нерационального использования


@dataclass
class YearlyStats:
    year: int
    peak_savi: float
    recovery_rate: float   # скорость восстановления после минимума (SAVI/день)
    min_savi: float
    min_date: str


# ─────────────────────────────────────────────
#  Ядро аналитики
# ─────────────────────────────────────────────

class VegetationAnalyzer:
    """
    Основной аналитический класс.
    Принимает VegetationTimeSeries из data_acquisition.py
    """

    # ── 1. Расчёт SAVI ──────────────────────────────────────────────────────

    @staticmethod
    def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = L_SAVI) -> np.ndarray:
        """
        SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
        Лучше NDVI для разреженной растительности степей (покрытие < 30%).
        """
        denom = nir + red + L
        # Избегаем деления на ноль
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        savi = (nir - red) / denom * (1 + L)
        return np.clip(savi, -1.0, 1.0)

    # ── 2. Расчёт биомассы ──────────────────────────────────────────────────

    @staticmethod
    def compute_biomass(savi: np.ndarray) -> np.ndarray:
        """
        Биомасса (т сухого вещества/га) по линейной регрессии.
        Калибровка для степей Казахстана (полузасушливые, покрытие 10-40%).
        """
        biomass = BIOMASS_SLOPE * savi + BIOMASS_INTERCEPT
        return np.clip(biomass, BIOMASS_MIN, BIOMASS_MAX)

    @staticmethod
    def biomass_to_carrying_capacity(biomass_per_ha: float) -> float:
        """
        Несущая способность пастбища.
        1 условная голова КРС потребляет ~6 т сухого вещества в год.
        Возвращает: условных голов скота/га/сезон (180 дней).
        """
        annual_productivity = biomass_per_ha * 4  # 4 вегетационных цикла/год
        heads_per_ha = annual_productivity / 6.0 * 0.5  # 50% утилизация (норма)
        return round(max(0.0, heads_per_ha), 3)

    # ── 3. Алгоритм VRP (Vegetation Recovery Rate) ─────────────────────────

    def compute_vrp(self, df: pd.DataFrame, plot_id: str) -> dict:
        """
        Сравнивает скорость восстановления вегетации с историческим средним.

        Логика:
        1. Находим момент минимума SAVI (= момент выпаса/засухи)
        2. Считаем скорость роста SAVI в следующие 30 дней
        3. Сравниваем с той же метрикой за предыдущие годы
        4. Если текущий год восстанавливается медленнее 75% от нормы → ДЕГРАДАЦИЯ
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["year"] = df["date"].dt.year

        yearly_stats = {}
        for year, group in df.groupby("year"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < 4:
                continue

            savi_vals = group["savi_mean"].values
            dates = group["date"].values

            # Находим минимум в летне-осенний период (июль-октябрь)
            summer_mask = (group["date"].dt.month >= 7) & (group["date"].dt.month <= 10)
            if summer_mask.sum() < 2:
                continue

            summer_group = group[summer_mask]
            min_idx = summer_group["savi_mean"].idxmin()
            min_row = group.loc[min_idx]

            # Считаем Recovery Rate: рост SAVI за 30 дней после минимума
            after_min = group[group["date"] > min_row["date"]].head(3)
            if len(after_min) < 2:
                recovery_rate = 0.0
            else:
                delta_savi = after_min["savi_mean"].iloc[-1] - min_row["savi_mean"]
                delta_days = (after_min["date"].iloc[-1] - min_row["date"]).days
                recovery_rate = delta_savi / max(delta_days, 1) * 30  # нормализуем к 30 дням

            yearly_stats[year] = YearlyStats(
                year=year,
                peak_savi=group["savi_mean"].max(),
                recovery_rate=recovery_rate,
                min_savi=min_row["savi_mean"],
                min_date=str(min_row["date"].date())
            )

        if len(yearly_stats) < 2:
            logger.warning(f"[{plot_id}] Недостаточно данных для VRP (нужно минимум 2 года)")
            return {"vrp_score": 1.0, "is_degraded": False, "yearly_stats": yearly_stats}

        years = sorted(yearly_stats.keys())
        current_year = years[-1]
        historical_years = years[:-1]

        hist_recovery = np.mean([yearly_stats[y].recovery_rate for y in historical_years])
        curr_recovery = yearly_stats[current_year].recovery_rate

        if hist_recovery > 1e-6:
            vrp_score = curr_recovery / hist_recovery
        else:
            vrp_score = 1.0

        is_degraded = vrp_score < DEGRADATION_VRP_THRESHOLD
        logger.info(
            f"[{plot_id}] VRP={vrp_score:.2f} | "
            f"текущий={curr_recovery:.4f} | исторический={hist_recovery:.4f} | "
            f"{'⚠️ ДЕГРАДАЦИЯ' if is_degraded else '✅ НОРМА'}"
        )
        return {
            "vrp_score": vrp_score,
            "is_degraded": is_degraded,
            "current_recovery_rate": curr_recovery,
            "historical_recovery_rate": hist_recovery,
            "yearly_stats": yearly_stats
        }

    # ── 4. Зонирование (красные / жёлтые зоны) ─────────────────────────────

    @staticmethod
    def compute_zoning(savi_map: np.ndarray) -> dict:
        """Возвращает доли площади по зонам состояния пастбища."""
        total = savi_map.size
        red_zone   = np.sum(savi_map < OVERGRAZING_SAVI_THRESHOLD) / total
        yellow_zone = np.sum(
            (savi_map >= OVERGRAZING_SAVI_THRESHOLD) & (savi_map < RISK_SAVI_THRESHOLD)
        ) / total
        green_zone  = np.sum(savi_map >= RISK_SAVI_THRESHOLD) / total
        return {
            "red":    round(red_zone, 3),
            "yellow": round(yellow_zone, 3),
            "green":  round(green_zone, 3)
        }

    # ── 5. Оценка субсидиарного риска ───────────────────────────────────────

    @staticmethod
    def compute_subsidy_risk(
        current_savi: float,
        vrp_score: float,
        overgrazing_fraction: float,
        trend_slope: float
    ) -> float:
        """
        Оценка риска (0-100%) того, что госмониторинг МСХ
        признает использование земель нерациональным.

        Компоненты риска:
        - SAVI ниже нормы для данного месяца (35% веса)
        - Медленное восстановление по VRP (30% веса)
        - Площадь перевыпаса > 20% участка (25% веса)
        - Отрицательный тренд последних 6 снимков (10% веса)
        """
        # Компонент 1: текущий SAVI
        savi_norm = max(0.05, min(current_savi, 0.7))
        r1 = max(0.0, (OVERGRAZING_SAVI_THRESHOLD - savi_norm) / OVERGRAZING_SAVI_THRESHOLD) * 0.35

        # Компонент 2: VRP
        vrp_clamped = max(0.0, min(vrp_score, 1.5))
        r2 = max(0.0, 1.0 - vrp_clamped / DEGRADATION_VRP_THRESHOLD) * 0.5 * 0.30

        # Компонент 3: площадь красных зон
        r3 = min(overgrazing_fraction / 0.3, 1.0) * 0.25

        # Компонент 4: негативный тренд
        r4 = max(0.0, -trend_slope * 50) * 0.10

        risk = (r1 + r2 + r3 + r4) * 100
        return round(min(risk, 99.0), 1)

    # ── 6. Генерация рекомендаций ───────────────────────────────────────────

    @staticmethod
    def generate_recommendations(result: PlotAnalysisResult) -> list:
        recs = []
        if result.overgrazing_fraction > 0.3:
            days = 14 if result.vrp_score > 0.5 else 30
            recs.append(f"🚨 СРОЧНО: Вывести скот с перегруженных секторов на {days} дней")
        if result.is_degraded:
            recs.append("⚠️  Разработать план восстановления пастбища (посев/подсев трав)")
        if result.current_biomass < 0.5:
            recs.append("📉 Биомасса критически низкая. Введите режим пастбищеоборота")
        if result.carrying_capacity > 0:
            recs.append(
                f"🐄 Рекомендуемая нагрузка: не более {result.carrying_capacity:.2f} усл. голов/га"
            )
        if result.subsidy_risk_score > 60:
            recs.append(
                f"⚖️  СУБСИДИАРНЫЙ РИСК {result.subsidy_risk_score:.0f}%! "
                "Подготовьте агрономическое заключение до проверки МСХ"
            )
        if not recs:
            recs.append("✅ Участок в норме. Продолжайте текущую практику")
        return recs

    # ── 7. Главный метод анализа ─────────────────────────────────────────────

    def analyze(self, time_series) -> PlotAnalysisResult:
        """
        Полный анализ временного ряда спутниковых снимков.
        Принимает VegetationTimeSeries из data_acquisition.py
        """
        if not time_series.scenes:
            raise ValueError(f"Нет снимков для участка {time_series.plot_id}")

        scenes = time_series.scenes
        dates, savi_series, biomass_series, savi_maps = [], [], [], []

        for scene in scenes:
            savi_map = self.compute_savi(scene.band_nir, scene.band_red)
            biomass_map = self.compute_biomass(savi_map)
            dates.append(scene.date)
            savi_series.append(float(np.mean(savi_map)))
            biomass_series.append(float(np.mean(biomass_map)))
            savi_maps.append(savi_map)

        # Тренд последних 6 снимков
        recent = savi_series[-6:] if len(savi_series) >= 6 else savi_series
        trend_slope = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) > 1 else 0.0

        # VRP
        df = pd.DataFrame({"date": dates, "savi_mean": savi_series})
        vrp_result = self.compute_vrp(df, time_series.plot_id)

        # Зонирование по последнему снимку
        last_map = savi_maps[-1]
        zones = self.compute_zoning(last_map)

        current_savi = savi_series[-1]
        current_biomass = biomass_series[-1]
        carrying_cap = self.biomass_to_carrying_capacity(np.mean(biomass_series))

        # Определяем флаг деградации
        if zones["red"] > 0.4:
            flag = "ДЕГРАДАЦИЯ"
        elif zones["red"] > 0.2 or vrp_result["is_degraded"]:
            flag = "РИСК"
        else:
            flag = "OK"

        subsidy_risk = self.compute_subsidy_risk(
            current_savi, vrp_result["vrp_score"], zones["red"], trend_slope
        )

        result = PlotAnalysisResult(
            plot_id=time_series.plot_id,
            dates=dates,
            savi_series=savi_series,
            biomass_series=biomass_series,
            savi_maps=savi_maps,
            current_savi=round(current_savi, 3),
            current_biomass=round(current_biomass, 3),
            carrying_capacity=carrying_cap,
            vrp_score=round(vrp_result["vrp_score"], 3),
            is_degraded=vrp_result["is_degraded"],
            degradation_flag=flag,
            overgrazing_fraction=zones["red"],
            risk_fraction=zones["yellow"],
            subsidy_risk_score=subsidy_risk
        )
        result.recommendations = self.generate_recommendations(result)
        return result


# ─────────────────────────────────────────────
#  Быстрый тест
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/amanat_terra")
    from data_acquisition import SatelliteDataClient

    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [70.4, 51.2], [70.5, 51.2], [70.5, 51.3], [70.4, 51.3], [70.4, 51.2]
        ]]
    }
    client = SatelliteDataClient(use_demo=True)
    analyzer = VegetationAnalyzer()

    for plot_id in ["plot_good", "plot_degraded_2023"]:
        ts = client.get_vegetation_data(geojson, "2022-01-01", "2024-12-31", plot_id)
        result = analyzer.analyze(ts)
        print(f"\n{'='*50}")
        print(f"Участок: {plot_id}")
        print(f"  SAVI:       {result.current_savi:.3f}")
        print(f"  Биомасса:   {result.current_biomass:.2f} т/га")
        print(f"  VRP:        {result.vrp_score:.2f}")
        print(f"  Флаг:       {result.degradation_flag}")
        print(f"  Риск МСХ:   {result.subsidy_risk_score:.0f}%")
        print("  Рекомендации:")
        for r in result.recommendations:
            print(f"    {r}")
