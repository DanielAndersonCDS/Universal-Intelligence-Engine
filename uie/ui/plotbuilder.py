"""CHARTS tab: pick a chart type and columns, draw from the open dataset, export PNG or SVG."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)
from ..core import plots
from .charts import EvidenceChart

HINTS = {
    "hist": "X = numeric column. Overlay fits the best simple distribution (normal, lognormal, exponential, gamma). Log X plots ln(x) for skewed positive data.",
    "scatter": "X and Y = numeric columns. Overlay fits the best curves (linear, quadratic, cubic, exponential, power, logarithmic).",
    "line": "X = date/time column. Y = numeric column (mean per period) or none for the row count.",
    "bar": "Group = category column. Y = numeric column (mean) or none for the row count.",
    "box": "Group = category column, Y = numeric column. Whiskers are Tukey (1.5 x IQR).",
    "heat": "Pearson correlation of up to 20 numeric columns. Blue negative, red positive.",
    "qq": "X = numeric column. Overlay tests the log of X for normality (lognormal check). Points on the line = normal.",
    "acf": "X = date/time column, Y = numeric column or none. Detrended autocorrelation; dashed lines are the 95% noise band.",
}


class PlotBuilder(QWidget):
    def __init__(self, engine_fn, run, status, colors):
        super().__init__()
        self.engine_fn, self.run, self.status = engine_fn, run, status
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.kind = QComboBox()
        for k, v in plots.KINDS.items():
            self.kind.addItem(v, k)
        self.x, self.y, self.g = QComboBox(), QComboBox(), QComboBox()
        self.bins = QSpinBox(); self.bins.setRange(5, 200); self.bins.setValue(40); self.bins.setPrefix("bins ")
        self.overlay = QCheckBox("Overlay fit")
        self.logx = QCheckBox("Log X (histogram)")
        go = QPushButton("DRAW"); go.setProperty("accent", True); go.clicked.connect(self.draw)
        for w in (self.kind, self.x, self.y, self.g, self.bins, self.overlay, self.logx, go):
            row.addWidget(w)
        lay.addLayout(row)
        self.hint = QLabel(""); self.hint.setObjectName("Dim"); self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.chart = EvidenceChart()
        self.chart.apply_theme({"colors": colors})
        self.chart.show_message("Choose a chart type and columns, then DRAW.")
        lay.addWidget(self.chart, 1)
        self.sql_lbl = QLabel(""); self.sql_lbl.setObjectName("Dim"); self.sql_lbl.setWordWrap(True)
        self.sql_lbl.setTextInteractionFlags(self.sql_lbl.textInteractionFlags() | self.sql_lbl.textInteractionFlags().TextSelectableByMouse)
        lay.addWidget(self.sql_lbl)
        bar = QHBoxLayout(); bar.addStretch(1)
        for ext in ("png", "svg"):
            b = QPushButton(f"EXPORT {ext.upper()}"); b.clicked.connect(lambda _=False, e=ext: self.export(e)); bar.addWidget(b)
        lay.addLayout(bar)
        self.kind.currentIndexChanged.connect(self._hint)
        self._hint()

    def set_colors(self, colors):
        self.chart.apply_theme({"colors": colors})

    def set_columns(self, schema):
        for combo, none in ((self.x, "X: choose"), (self.y, "Y: none / choose"), (self.g, "Group: choose")):
            keep = combo.currentData(); combo.clear(); combo.addItem(none, None)
            for n, _t in schema:
                combo.addItem(f"{none.split(':')[0]}: {n}", n)
            i = combo.findData(keep); combo.setCurrentIndex(max(i, 0))

    def _hint(self):
        self.hint.setText(HINTS[self.kind.currentData()])

    def draw(self):
        eng = self.engine_fn()
        if not eng.src:
            return self.status("OPEN A DATASET FIRST")
        kind, x, y, g, bins, ov, lg = self.kind.currentData(), self.x.currentData(), self.y.currentData(), self.g.currentData(), self.bins.value(), self.overlay.isChecked(), self.logx.isChecked()
        self.chart.show_message("Drawing...")

        def done(res):
            spec, rows, sql = res
            self.chart.show_data(spec, rows)
            self.sql_lbl.setText("SQL: " + sql[:600])
            self.status(f"CHART: {plots.KINDS[kind]}, {len(rows):,} points")
        self.run(lambda p: plots.build_chart(eng.src.con.cursor(), kind, x, y, g, bins, ov, log_x=lg), done, lambda e: (self.chart.show_message(e), self.status("CHART ERROR: " + e)))

    def export(self, ext):
        path, _ = QFileDialog.getSaveFileName(self, f"Export {ext.upper()}", f"chart.{ext}")
        if not path:
            return
        try:
            self.chart.export_image(path)
            self.status(f"EXPORTED {path}")
        except Exception as e:
            self.status("EXPORT FAILED: " + str(e))
