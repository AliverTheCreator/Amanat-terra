"""
Amanat-Terra | Модуль 5: Хранение данных (CSV / PostgreSQL)
===========================================================
Сохраняет результаты анализа в CSV или PostgreSQL.

Использование:
    from data_storage import DataStorage
    storage = DataStorage(mode="csv", path="results/")
    storage.save(result)
    df = storage.load_all()
"""

import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger("amanat_terra.storage")


class DataStorage:
    """
    Сохраняет результаты анализа участков.
    mode: "csv" — CSV-файлы (для хакатона)
          "postgres" — PostgreSQL (для production)
    """

    def __init__(self, mode: str = "csv", path: str = "./data/", db_url: str = ""):
        self.mode = mode
        self.path = path
        os.makedirs(path, exist_ok=True)

        if mode == "postgres":
            self._init_postgres(db_url)
        else:
            logger.info(f"📂 Хранилище: CSV в '{path}'")

    def _init_postgres(self, db_url: str):
        try:
            import psycopg2
            from sqlalchemy import create_engine, text
            self.engine = create_engine(db_url)
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS plot_analysis (
                        id          SERIAL PRIMARY KEY,
                        plot_id     VARCHAR(100),
                        date        TIMESTAMP,
                        current_savi FLOAT,
                        current_biomass FLOAT,
                        carrying_capacity FLOAT,
                        vrp_score   FLOAT,
                        degradation_flag VARCHAR(20),
                        overgrazing_fraction FLOAT,
                        subsidy_risk_score FLOAT,
                        recommendations TEXT,
                        raw_json    JSONB
                    );
                    CREATE INDEX IF NOT EXISTS idx_plot_id ON plot_analysis(plot_id);
                    CREATE INDEX IF NOT EXISTS idx_date ON plot_analysis(date);
                """))
                conn.commit()
            logger.info("✅ PostgreSQL подключён")
        except ImportError:
            raise ImportError("Установите: pip install psycopg2 sqlalchemy")

    def save(self, result) -> str:
        """Сохраняет результат анализа."""
        record = {
            "plot_id":              result.plot_id,
            "date":                 datetime.now().isoformat(),
            "current_savi":         result.current_savi,
            "current_biomass":      result.current_biomass,
            "carrying_capacity":    result.carrying_capacity,
            "vrp_score":            result.vrp_score,
            "degradation_flag":     result.degradation_flag,
            "overgrazing_fraction": result.overgrazing_fraction,
            "risk_fraction":        result.risk_fraction,
            "subsidy_risk_score":   result.subsidy_risk_score,
            "recommendations":      " | ".join(result.recommendations),
            "n_scenes":             len(result.dates) if result.dates else 0,
            "date_start":           result.dates[0] if result.dates else "",
            "date_end":             result.dates[-1] if result.dates else "",
        }

        if self.mode == "csv":
            return self._save_csv(record, result)
        else:
            return self._save_postgres(record)

    def _save_csv(self, record: dict, result) -> str:
        # 1. Сводная таблица участков
        summary_path = os.path.join(self.path, "plots_summary.csv")
        is_new = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(record)

        # 2. Временной ряд SAVI для участка
        if result.dates:
            ts_path = os.path.join(self.path, f"ts_{result.plot_id}.csv")
            df_ts = pd.DataFrame({
                "date":    result.dates,
                "savi":    result.savi_series,
                "biomass": result.biomass_series
            })
            df_ts.to_csv(ts_path, index=False, encoding="utf-8")
            logger.info(f"💾 Временной ряд сохранён: {ts_path}")

        logger.info(f"💾 Сводка сохранена: {summary_path}")
        return summary_path

    def _save_postgres(self, record: dict) -> str:
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_sql("plot_analysis", self.engine, if_exists="append", index=False)
        logger.info(f"💾 PostgreSQL: запись для {record['plot_id']} сохранена")
        return "postgres"

    def load_all(self) -> pd.DataFrame:
        """Загружает все сохранённые результаты анализа."""
        if self.mode == "csv":
            path = os.path.join(self.path, "plots_summary.csv")
            if not os.path.exists(path):
                return pd.DataFrame()
            return pd.read_csv(path, encoding="utf-8")
        else:
            return pd.read_sql("SELECT * FROM plot_analysis ORDER BY date DESC", self.engine)

    def load_timeseries(self, plot_id: str) -> pd.DataFrame:
        """Загружает временной ряд по конкретному участку."""
        if self.mode == "csv":
            path = os.path.join(self.path, f"ts_{plot_id}.csv")
            if not os.path.exists(path):
                return pd.DataFrame()
            return pd.read_csv(path, parse_dates=["date"], encoding="utf-8")
        else:
            return pd.read_sql(
                f"SELECT * FROM plot_timeseries WHERE plot_id = '{plot_id}' ORDER BY date",
                self.engine
            )


if __name__ == "__main__":
    print("Модуль DataStorage готов к использованию.")
    print("Пример:")
    print("  storage = DataStorage(mode='csv', path='./data/')")
    print("  storage.save(result)")
    print("  df = storage.load_all()")
