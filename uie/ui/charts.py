from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget
from ..core.util import to_ts


class EvidenceChart(QWidget):
    """Renders a finding's evidence query. Specs: line (time), bar, bins (histogram), scatter."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack)
        self.msg = QLabel("Select a finding to see its evidence.")
        self.msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.msg.setWordWrap(True)
        self.msg.setObjectName("Dim")
        self.time_plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self.cat_plot = pg.PlotWidget()
        for w in (self.msg, self.time_plot, self.cat_plot):
            self.stack.addWidget(w)
        self.c: dict = {}
        self._last = None

    def apply_theme(self, theme: dict):
        self.c = theme["colors"]
        for p in (self.time_plot, self.cat_plot):
            p.setBackground(self.c["panel"])
            for ax in ("left", "bottom"):
                a = p.getAxis(ax)
                a.setPen(pg.mkPen(self.c["border"]))
                a.setTextPen(pg.mkPen(self.c["text_dim"]))
            p.showGrid(x=True, y=True, alpha=0.15)
        if self._last:
            self.show_data(*self._last)

    def show_message(self, text: str):
        self._last = None
        self.msg.setText(text)
        self.stack.setCurrentWidget(self.msg)

    def _labels(self, p, spec):
        col = self.c.get("text_dim", "#888")
        p.setLabel("bottom", spec.get("xlabel", ""), color=col)
        p.setLabel("left", spec.get("ylabel", ""), color=col)

    def show_data(self, spec: dict, rows: list):
        if not rows:
            return self.show_message("The evidence query returned no rows.")
        self._last = (spec, rows)
        kind, c = spec["kind"], self.c
        accent = pg.mkPen(c["accent"], width=2)
        if kind == "line":
            p = self.time_plot
            p.clear()
            xs = np.array([to_ts(r[0]) for r in rows], float)
            ys = np.array([np.nan if r[1] is None else float(r[1]) for r in rows])
            p.plot(xs, ys, pen=accent, connect="finite")
            if spec.get("marker_ts"):
                p.addItem(pg.InfiniteLine(pos=spec["marker_ts"], angle=90,
                                          pen=pg.mkPen(c["warning"], width=1, style=Qt.PenStyle.DashLine)))
        else:
            p = self.cat_plot
            p.clear()
            ax = p.getAxis("bottom")
            ax.setTicks(None)
            if kind == "bar":
                rows = rows[:12]
                ys = np.array([float(r[1] or 0) for r in rows])
                p.addItem(pg.BarGraphItem(x=np.arange(len(rows)), height=ys, width=0.7, brush=c["accent"], pen=None))
                ax.setTicks([[(i, str(r[0])[:10]) for i, r in enumerate(rows)]])
            elif kind == "bins":
                w, mn = spec["w"], spec["mn"]
                xs = np.array([mn + (r[0] + 0.5) * w for r in rows], float)
                ys = np.log10(np.array([float(r[1]) for r in rows]) + 1)
                p.addItem(pg.BarGraphItem(x=xs, height=ys, width=w * 0.9, brush=c["accent"], pen=None))
                for v in spec.get("lines", []):
                    p.addItem(pg.InfiniteLine(pos=v, angle=90, pen=pg.mkPen(c["warning"], style=Qt.PenStyle.DashLine)))
            elif kind == "scatter":
                xs = np.array([float(r[0]) for r in rows])
                ys = np.array([float(r[1]) for r in rows])
                p.addItem(pg.ScatterPlotItem(x=xs, y=ys, size=4, pen=None, brush=pg.mkBrush(c["accent"] + "99")))
        self._labels(p, spec)
        self.stack.setCurrentWidget(p)
