"""
Amanat-Terra | Модуль 3: Генератор PDF-отчётов для МСХ
=====================================================
Создаёт профессиональный отчёт «Соответствия» (Compliance Report)
на русском языке для подачи в акимат / МСХ.

Использование:
    from report_generator import ComplianceReportGenerator
    gen = ComplianceReportGenerator()
    gen.generate(analysis_result, output_path="report.pdf")
"""

import io
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

logger = logging.getLogger("amanat_terra.report")

# ─────────────────────────────────────────────
#  Цветовая палитра
# ─────────────────────────────────────────────
C_GREEN  = colors.HexColor("#2ecc71")
C_YELLOW = colors.HexColor("#f39c12")
C_RED    = colors.HexColor("#e74c3c")
C_BLUE   = colors.HexColor("#2980b9")
C_DARK   = colors.HexColor("#2c3e50")
C_LIGHT  = colors.HexColor("#ecf0f1")
C_WHITE  = colors.white


def _status_color(flag: str):
    return {"OK": C_GREEN, "РИСК": C_YELLOW, "ДЕГРАДАЦИЯ": C_RED}.get(flag, C_BLUE)


# ─────────────────────────────────────────────
#  Генератор графиков
# ─────────────────────────────────────────────

class ChartGenerator:

    @staticmethod
    def savi_timeseries(dates, savi_values, title="Динамика SAVI", width_cm=16, height_cm=6) -> io.BytesIO:
        """График временного ряда SAVI с зонами."""
        fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
        x = range(len(dates))
        ax.fill_between(x, 0, 0.18, alpha=0.15, color="red",    label="Перевыпас (<0.18)")
        ax.fill_between(x, 0.18, 0.25, alpha=0.15, color="orange", label="Риск (0.18-0.25)")
        ax.fill_between(x, 0.25, 1.0, alpha=0.08, color="green", label="Норма (>0.25)")
        ax.plot(x, savi_values, "b-o", markersize=3, linewidth=1.5, label="SAVI")
        ax.axhline(0.18, color="red",    linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0.25, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)

        # Показываем только каждую 5-ю дату
        tick_step = max(1, len(dates) // 8)
        ax.set_xticks(list(x)[::tick_step])
        ax.set_xticklabels([dates[i][:7] for i in range(0, len(dates), tick_step)],
                           rotation=35, fontsize=7)
        ax.set_ylabel("SAVI", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(0, max(max(savi_values) * 1.2, 0.5))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def savi_heatmap(savi_map: np.ndarray, title="Карта вегетации", width_cm=10, height_cm=9) -> io.BytesIO:
        """2D тепловая карта SAVI с красными зонами перевыпаса."""
        fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))

        # Кастомная цветовая шкала: красный → жёлтый → зелёный
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pasture", [(0, "#e74c3c"), (0.3, "#f39c12"), (0.6, "#f1c40f"), (1.0, "#27ae60")]
        )
        im = ax.imshow(savi_map, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("SAVI", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # Обводим критические зоны
        ax.contour(savi_map, levels=[0.18], colors=["red"], linewidths=1.5, alpha=0.8)
        ax.contour(savi_map, levels=[0.25], colors=["orange"], linewidths=1.0, alpha=0.6)

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Запад → Восток", fontsize=7)
        ax.set_ylabel("Юг → Север", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def subsidy_risk_gauge(risk_score: float, width_cm=8, height_cm=6) -> io.BytesIO:
        """Спидометр субсидиарного риска."""
        fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54),
                               subplot_kw={"aspect": "equal"})
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.axis("off")

        # Дуга фона
        theta = np.linspace(np.pi, 0, 200)
        r = 1.0
        for i, (lo, hi, col) in enumerate([(0, 30, "#2ecc71"), (30, 60, "#f39c12"), (60, 100, "#e74c3c")]):
            t_start = np.pi - lo / 100 * np.pi
            t_end   = np.pi - hi / 100 * np.pi
            t_range = np.linspace(t_start, t_end, 50)
            xs = r * np.cos(t_range)
            ys = r * np.sin(t_range)
            ax.fill_between(xs, 0, ys, alpha=0.3, color=col)
            ax.plot(xs, ys, color=col, linewidth=3)

        # Стрелка
        angle = np.pi - (risk_score / 100) * np.pi
        ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))
        ax.plot(0, 0, "ko", markersize=6)

        # Текст
        ax.text(0, -0.15, f"{risk_score:.0f}%", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="#e74c3c" if risk_score > 60 else ("#f39c12" if risk_score > 30 else "#2ecc71"))
        ax.set_title("Субсидиарный риск МСХ", fontsize=8, fontweight="bold")

        for pct, label in [(0, "0"), (50, "50"), (100, "100%")]:
            a = np.pi - pct / 100 * np.pi
            ax.text(1.15 * np.cos(a), 1.15 * np.sin(a), label, ha="center", va="center", fontsize=6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def zone_pie(red, yellow, green, width_cm=7, height_cm=5) -> io.BytesIO:
        """Круговая диаграмма зон состояния пастбища."""
        fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
        vals, lbls, clrs = [], [], []
        for v, l, c in [(red, f"Перевыпас\n{red:.0%}", "#e74c3c"),
                        (yellow, f"Риск\n{yellow:.0%}", "#f39c12"),
                        (green, f"Норма\n{green:.0%}", "#2ecc71")]:
            if v > 0.001:
                vals.append(v); lbls.append(l); clrs.append(c)
        ax.pie(vals, labels=lbls, colors=clrs, autopct="", startangle=90,
               textprops={"fontsize": 7}, wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        ax.set_title("Зонирование участка", fontsize=8, fontweight="bold")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf


# ─────────────────────────────────────────────
#  Генератор PDF
# ─────────────────────────────────────────────

class ComplianceReportGenerator:
    """Генерирует PDF Compliance Report для МСХ/акимата."""

    def __init__(self):
        self.charts = ChartGenerator()
        self._setup_styles()

    def _setup_styles(self):
        base = getSampleStyleSheet()
        self.styles = {
            "title": ParagraphStyle("title", parent=base["Title"],
                                    fontSize=18, textColor=C_DARK, spaceAfter=4, alignment=TA_CENTER),
            "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                       fontSize=11, textColor=C_BLUE, spaceAfter=2, alignment=TA_CENTER),
            "heading": ParagraphStyle("heading", parent=base["Heading2"],
                                      fontSize=12, textColor=C_DARK, spaceBefore=10, spaceAfter=4,
                                      borderPad=4, backColor=C_LIGHT),
            "body": ParagraphStyle("body", parent=base["Normal"],
                                   fontSize=9, leading=13, spaceAfter=4),
            "small": ParagraphStyle("small", parent=base["Normal"],
                                    fontSize=8, textColor=colors.grey),
            "rec": ParagraphStyle("rec", parent=base["Normal"],
                                  fontSize=9, leading=13, leftIndent=8, spaceAfter=3),
        }

    def _header_block(self, plot_id: str, flag: str, generated_date: str) -> list:
        flag_color = _status_color(flag)
        story = [
            Paragraph("🌿 AMANAT-TERRA", self.styles["title"]),
            Paragraph("Отчёт о состоянии пастбищных угодий", self.styles["subtitle"]),
            Paragraph(f"Система мониторинга земель Республики Казахстан", self.styles["subtitle"]),
            HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8),
        ]
        # Статус-баннер
        status_table = Table([[
            Paragraph(f"Кадастровый участок: <b>{plot_id}</b>", self.styles["body"]),
            Paragraph(f"Дата отчёта: <b>{generated_date}</b>", self.styles["body"]),
            Paragraph(f"Статус: <b>{flag}</b>", ParagraphStyle(
                "status", parent=self.styles["body"], textColor=flag_color, fontSize=11
            ))
        ]], colWidths=["40%", "30%", "30%"])
        status_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT]),
            ("BOX", (0, 0), (-1, -1), 0.5, C_BLUE),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(status_table)
        story.append(Spacer(1, 0.3 * cm))
        return story

    def _metrics_table(self, result) -> list:
        story = [Paragraph("📊 Ключевые показатели", self.styles["heading"])]
        data = [
            ["Показатель", "Значение", "Норма", "Оценка"],
            ["SAVI (текущий)", f"{result.current_savi:.3f}",
             "> 0.25", "✅" if result.current_savi >= 0.25 else "⚠️" if result.current_savi >= 0.18 else "🔴"],
            ["Биомасса", f"{result.current_biomass:.2f} т/га",
             "0.5–4.5", "✅" if result.current_biomass >= 0.5 else "🔴"],
            ["Нес. способность", f"{result.carrying_capacity:.3f} гол/га",
             "регион. норма", "—"],
            ["VRP (восстановление)", f"{result.vrp_score:.2f}",
             "> 0.75", "✅" if result.vrp_score >= 0.75 else "🔴"],
            ["Зона перевыпаса", f"{result.overgrazing_fraction:.0%}",
             "< 20%", "✅" if result.overgrazing_fraction < 0.2 else "🔴"],
            ["Субсидиарный риск", f"{result.subsidy_risk_score:.0f}%",
             "< 30%", "✅" if result.subsidy_risk_score < 30 else "⚠️" if result.subsidy_risk_score < 60 else "🔴"],
        ]
        t = Table(data, colWidths=["35%", "20%", "20%", "25%"])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("BOX", (0, 0), (-1, -1), 0.5, C_DARK),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))
        return story

    def _recommendations_block(self, result) -> list:
        story = [Paragraph("📋 Рекомендации агронома", self.styles["heading"])]
        for rec in result.recommendations:
            story.append(Paragraph(rec, self.styles["rec"]))
        story.append(Spacer(1, 0.2 * cm))
        # Правовая справка
        if result.subsidy_risk_score > 50:
            story.append(Paragraph(
                "<b>⚖️ Правовая справка:</b> Согласно ст. 141 Земельного кодекса РК, "
                "нерациональное использование пастбищных угодий влечёт предупреждение "
                "и возможное изъятие участка. Настоящий отчёт рекомендуется предъявить "
                "в акимат как доказательство принятых мер.",
                self.styles["body"]
            ))
        return story

    def _footer_block(self) -> list:
        return [
            HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=4),
            Paragraph(
                "Отчёт сформирован системой Amanat-Terra на основе данных Sentinel-2 (ESA Copernicus). "
                "Данные предназначены для агрономического мониторинга. "
                "Не является официальным заключением государственных органов.",
                self.styles["small"]
            )
        ]

    def generate(self, result, output_path: str = "compliance_report.pdf",
                 plot_name: str = None) -> str:
        """
        Генерирует PDF-отчёт.
        result      — PlotAnalysisResult из analytics_engine.py
        output_path — путь для сохранения
        """
        plot_name = plot_name or result.plot_id
        generated_date = datetime.now().strftime("%d.%m.%Y %H:%M")
        logger.info(f"📄 Генерация отчёта для {plot_name}...")

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=1.8 * cm, leftMargin=1.8 * cm,
            topMargin=1.5 * cm, bottomMargin=1.5 * cm
        )
        story = []

        # ── Шапка ──────────────────────────────────────────────────────────
        story += self._header_block(plot_name, result.degradation_flag, generated_date)

        # ── Ключевые метрики ────────────────────────────────────────────────
        story += self._metrics_table(result)

        # ── Графики: тепловая карта + спидометр риска ──────────────────────
        story.append(Paragraph("🗺️  Визуализация состояния пастбища", self.styles["heading"]))
        chart_data = []

        if result.savi_maps:
            heatmap_buf = self.charts.savi_heatmap(result.savi_maps[-1],
                                                   title=f"Карта SAVI — {result.dates[-1]}")
            chart_data.append(Image(heatmap_buf, width=10 * cm, height=9 * cm))

        gauge_buf = self.charts.subsidy_risk_gauge(result.subsidy_risk_score)
        pie_buf   = self.charts.zone_pie(result.overgrazing_fraction,
                                         result.risk_fraction,
                                         1 - result.overgrazing_fraction - result.risk_fraction)
        right_col = [Image(gauge_buf, width=8 * cm, height=6 * cm),
                     Spacer(1, 0.3 * cm),
                     Image(pie_buf, width=7 * cm, height=5 * cm)]
        from reportlab.platypus import KeepInFrame

        if chart_data:
            chart_table = Table([chart_data + [right_col]], colWidths=["52%", "48%"])
            chart_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(chart_table)
        else:
            row = [Image(gauge_buf, width=9 * cm, height=7 * cm),
                   Image(pie_buf, width=8 * cm, height=6 * cm)]
            ct = Table([row], colWidths=["50%", "50%"])
            story.append(ct)

        story.append(Spacer(1, 0.3 * cm))

        # ── Временной ряд SAVI ──────────────────────────────────────────────
        if result.dates and result.savi_series:
            story.append(Paragraph("📈 Динамика вегетации за период", self.styles["heading"]))
            ts_buf = self.charts.savi_timeseries(result.dates, result.savi_series,
                                                  title=f"SAVI по времени — {plot_name}")
            story.append(Image(ts_buf, width=17 * cm, height=6.5 * cm))
            story.append(Spacer(1, 0.3 * cm))

        # ── Рекомендации ────────────────────────────────────────────────────
        story += self._recommendations_block(result)

        # ── Подвал ──────────────────────────────────────────────────────────
        story += self._footer_block()

        doc.build(story)
        logger.info(f"✅ Отчёт сохранён: {output_path}")
        return output_path


# ─────────────────────────────────────────────
#  Быстрый тест
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/amanat_terra")
    from data_acquisition import SatelliteDataClient
    from analytics_engine import VegetationAnalyzer

    geojson = {"type": "Polygon", "coordinates": [[[70.4, 51.2], [70.5, 51.2],
                [70.5, 51.3], [70.4, 51.3], [70.4, 51.2]]]}

    client = SatelliteDataClient(use_demo=True)
    analyzer = VegetationAnalyzer()
    generator = ComplianceReportGenerator()

    for pid, name in [("plot_good", "Акмолинский р-н (норма)"),
                      ("plot_degraded_severe", "Карагандинский р-н (деградация)")]:
        ts = client.get_vegetation_data(geojson, "2022-01-01", "2024-12-31", pid)
        result = analyzer.analyze(ts)
        out = f"/home/claude/amanat_terra/{pid}_report.pdf"
        generator.generate(result, out, plot_name=name)
        print(f"📄 Создан отчёт: {out}")
