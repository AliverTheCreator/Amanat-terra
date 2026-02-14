"""
Amanat-Terra | Модуль 4: Streamlit Дашборд
==========================================
Запуск: streamlit run dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import io
from datetime import datetime, date

from data_acquisition import SatelliteDataClient
from analytics_engine import VegetationAnalyzer
from report_generator import ComplianceReportGenerator, ChartGenerator

# ─────────────────────────────────────────────
#  Конфигурация страницы
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Amanat-Terra | Мониторинг пастбищ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #2980b9;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .status-ok     { border-left-color: #2ecc71 !important; }
    .status-risk   { border-left-color: #f39c12 !important; }
    .status-bad    { border-left-color: #e74c3c !important; }
    .big-metric    { font-size: 2rem; font-weight: bold; }
    .sidebar-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Демо участки (реальные координаты РК)
# ─────────────────────────────────────────────

DEMO_PLOTS = {
    "Акмолинская обл. — Целиноградский р-н (норма)": {
        "id": "plot_good_akmola",
        "geojson": {"type": "Polygon", "coordinates": [[[71.4, 51.8], [71.6, 51.8], [71.6, 52.0], [71.4, 52.0], [71.4, 51.8]]]},
        "description": "Степная зона, хорошее состояние, ротация соблюдается"
    },
    "Карагандинская обл. — Улытауский р-н (деградация)": {
        "id": "plot_degraded_karaganda",
        "geojson": {"type": "Polygon", "coordinates": [[[67.2, 48.6], [67.4, 48.6], [67.4, 48.8], [67.2, 48.8], [67.2, 48.6]]]},
        "description": "Полузасушливая зона, признаки деградации, перевыпас"
    },
    "Южно-Казахстанская обл. — Сузакский р-н (средний риск)": {
        "id": "plot_risk_shymkent",
        "geojson": {"type": "Polygon", "coordinates": [[[68.5, 43.5], [68.7, 43.5], [68.7, 43.7], [68.5, 43.7], [68.5, 43.5]]]},
        "description": "Пастбища предгорий, умеренная нагрузка"
    },
}

# ─────────────────────────────────────────────
#  Sidebar — фильтры
# ─────────────────────────────────────────────

st.sidebar.markdown('<div class="sidebar-title">🌿 Amanat-Terra</div>', unsafe_allow_html=True)
st.sidebar.markdown("*Спутниковый мониторинг пастбищ РК*")
st.sidebar.divider()

st.sidebar.markdown("**📍 Выбор участка**")
plot_name = st.sidebar.selectbox("Участок:", list(DEMO_PLOTS.keys()))
plot_info = DEMO_PLOTS[plot_name]

st.sidebar.markdown("**📅 Временной диапазон**")
year_start = st.sidebar.slider("Начальный год:", 2021, 2023, 2022)
year_end   = st.sidebar.slider("Конечный год:",  2022, 2024, 2024)
start_date = f"{year_start}-01-01"
end_date   = f"{year_end}-12-31"

st.sidebar.markdown("**⚙️ Настройки**")
api_mode = st.sidebar.radio("Источник данных:", ["🎭 Demo (Демо)", "🛰️ Sentinel Hub (API)"])
use_demo = api_mode.startswith("🎭")

if not use_demo:
    st.sidebar.text_input("Sentinel Hub Client ID", type="password", key="sh_id")
    st.sidebar.text_input("Sentinel Hub Client Secret", type="password", key="sh_secret")

st.sidebar.divider()
run_analysis = st.sidebar.button("🚀 Запустить анализ", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  Основная страница
# ─────────────────────────────────────────────

st.title("🌿 Amanat-Terra — Мониторинг пастбищ")
st.caption(f"Участок: **{plot_name}** | {plot_info['description']}")

if not run_analysis:
    # Приветственный экран
    st.info("👈 Выберите участок и нажмите **Запустить анализ** для получения результатов.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🛰️ Спутниковые снимки", "Sentinel-2", "10м/пиксель")
    with col2:
        st.metric("📊 Индексы вегетации", "SAVI + VRP", "адаптировано для РК")
    with col3:
        st.metric("📄 Отчётность", "МСХ / Акимат", "формат PDF")

    # Карта участков
    st.subheader("📍 Демо участки на карте")
    map_df = pd.DataFrame([
        {"lat": 51.9, "lon": 71.5, "name": "Акмолинская (норма)"},
        {"lat": 48.7, "lon": 67.3, "name": "Карагандинская (деградация)"},
        {"lat": 43.6, "lon": 68.6, "name": "ЮКО (риск)"},
    ])
    st.map(map_df, latitude="lat", longitude="lon")

else:
    # ── Анализ ─────────────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="⏳ Получение спутниковых данных...")

    try:
        client_id = st.session_state.get("sh_id", "")
        client_secret = st.session_state.get("sh_secret", "")
        client = SatelliteDataClient(client_id, client_secret, use_demo=use_demo)
        progress_bar.progress(20, text="📡 Загрузка снимков...")

        ts = client.get_vegetation_data(
            plot_info["geojson"], start_date, end_date,
            plot_id=plot_info["id"]
        )
        progress_bar.progress(50, text="🧮 Расчёт SAVI и биомассы...")

        analyzer = VegetationAnalyzer()
        result = analyzer.analyze(ts)
        progress_bar.progress(80, text="📊 Генерация визуализаций...")

        charts = ChartGenerator()
        progress_bar.progress(100, text="✅ Готово!")
        progress_bar.empty()

        # ── Статус-баннер ────────────────────────────────────────────────────
        flag_colors = {"OK": "green", "РИСК": "orange", "ДЕГРАДАЦИЯ": "red"}
        flag_icons  = {"OK": "✅", "РИСК": "⚠️", "ДЕГРАДАЦИЯ": "🔴"}
        flag = result.degradation_flag
        st.markdown(
            f"<div style='background:{'#d5f5e3' if flag=='OK' else '#fef9e7' if flag=='РИСК' else '#fadbd8'};"
            f"border-left:6px solid {'#2ecc71' if flag=='OK' else '#f39c12' if flag=='РИСК' else '#e74c3c'};"
            f"padding:12px 20px;border-radius:8px;margin-bottom:12px;font-size:1.1rem;font-weight:600'>"
            f"{flag_icons[flag]} Статус участка: <b>{flag}</b></div>",
            unsafe_allow_html=True
        )

        # ── Метрики ──────────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🌿 SAVI", f"{result.current_savi:.3f}",
                  delta="✅ Норма" if result.current_savi >= 0.25 else "⚠️ Низкий")
        m2.metric("🌾 Биомасса", f"{result.current_biomass:.2f} т/га")
        m3.metric("🐄 Нес. способность", f"{result.carrying_capacity:.3f} гол/га")
        m4.metric("📉 VRP Восстановление", f"{result.vrp_score:.2f}",
                  delta="✅ Норма" if result.vrp_score >= 0.75 else "⚠️ Медленное")
        m5.metric("⚖️ Риск МСХ", f"{result.subsidy_risk_score:.0f}%",
                  delta="Высокий" if result.subsidy_risk_score > 60 else
                        "Средний" if result.subsidy_risk_score > 30 else "Низкий")

        st.divider()

        # ── Карты и графики ──────────────────────────────────────────────────
        col_map, col_right = st.columns([1.2, 0.8])

        with col_map:
            st.subheader("🗺️ Карта вегетации участка")
            if result.savi_maps:
                fig, ax = plt.subplots(figsize=(7, 6))
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "p", [(0, "#e74c3c"), (0.3, "#f39c12"), (0.6, "#f1c40f"), (1.0, "#27ae60")]
                )
                im = ax.imshow(result.savi_maps[-1], cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
                plt.colorbar(im, ax=ax, label="SAVI")
                ax.contour(result.savi_maps[-1], levels=[0.18], colors=["red"], linewidths=2)
                ax.contour(result.savi_maps[-1], levels=[0.25], colors=["orange"], linewidths=1.5)
                ax.set_title(f"SAVI карта — {result.dates[-1] if result.dates else 'нет данных'}",
                             fontweight="bold")
                ax.set_xlabel("Запад → Восток")
                ax.set_ylabel("Юг → Север")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.caption("🔴 Красный контур = перевыпас (SAVI < 0.18) | 🟠 Оранжевый = риск (< 0.25)")

        with col_right:
            st.subheader("⚖️ Субсидиарный риск МСХ")
            gauge_buf = charts.subsidy_risk_gauge(result.subsidy_risk_score)
            st.image(gauge_buf, use_container_width=True)

            st.subheader("🥧 Зонирование")
            pie_buf = charts.zone_pie(
                result.overgrazing_fraction, result.risk_fraction,
                max(0, 1 - result.overgrazing_fraction - result.risk_fraction)
            )
            st.image(pie_buf, use_container_width=True)

        # ── Временной ряд ────────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Динамика вегетации за период")

        if result.dates:
            df_ts = pd.DataFrame({
                "Дата": pd.to_datetime(result.dates),
                "SAVI": result.savi_series,
                "Биомасса (т/га)": result.biomass_series
            }).set_index("Дата")

            tab1, tab2 = st.tabs(["📊 График SAVI", "📋 Таблица данных"])
            with tab1:
                ts_buf = charts.savi_timeseries(result.dates, result.savi_series,
                                                 title=f"Динамика SAVI — {plot_name}")
                st.image(ts_buf, use_container_width=True)
            with tab2:
                st.dataframe(
                    df_ts.style.background_gradient(subset=["SAVI"], cmap="RdYlGn",
                                                     vmin=0.1, vmax=0.5),
                    use_container_width=True, height=300
                )

        # ── Рекомендации ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Рекомендации агронома")
        for rec in result.recommendations:
            icon = "🚨" if "СРОЧНО" in rec or "СУБСИД" in rec else \
                   "⚠️" if "риск" in rec.lower() or "критич" in rec.lower() else "✅"
            with st.container(border=True):
                st.markdown(rec)

        # ── Кнопка скачать отчёт ─────────────────────────────────────────────
        st.divider()
        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            if st.button("📄 Сформировать отчёт для МСХ", type="primary", use_container_width=True):
                with st.spinner("Генерация PDF..."):
                    import tempfile
                    gen = ComplianceReportGenerator()
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        gen.generate(result, tmp.name, plot_name=plot_name)
                        with open(tmp.name, "rb") as f:
                            pdf_bytes = f.read()

                    st.download_button(
                        label="⬇️ Скачать PDF отчёт",
                        data=pdf_bytes,
                        file_name=f"amanat_terra_{plot_info['id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        type="secondary"
                    )
        with col_note:
            st.info("📬 Отчёт сформирован согласно требованиям МСХ РК для подтверждения рационального использования пастбищных угодий.")

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Ошибка анализа: {str(e)}")
        st.exception(e)
