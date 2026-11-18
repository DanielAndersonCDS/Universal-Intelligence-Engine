from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget
from ..core import stats
from ..core.curvefit import predict
from ..core.util import to_ts


class EvidenceChart(QWidget):
    """Renders evidence queries and plot-builder output. Specs: line, bar, bins, hist (+fitted pdf), scatter (+fitted curves), box, heat, qq, acf."""

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
        self.time_plot.addLegend(offset=(10, 10))
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
        if kind == "multiline":
            p = self.time_plot
            p.clear()
            p.setLogMode(x=False, y=False)
            xs = np.array([to_ts(r[0]) for r in rows], float)
            names = spec.get("names", [])
            cols = [c["accent"], c["info"], c["warning"], c["positive"], c["text_dim"]]
            band = spec.get("band")
            curves = []
            for j in range(1, len(rows[0])):
                ys = np.array([np.nan if r[j] is None else float(r[j]) for r in rows])
                if spec.get("logy"):
                    ys = np.where(ys > 0, ys, np.nan)
                is_band = band and (j - 1) in band
                pen = pg.mkPen(cols[(j - 1) % len(cols)], width=1 if is_band else 2, style=Qt.PenStyle.DashLine if is_band else Qt.PenStyle.SolidLine)
                curves.append(p.plot(xs, np.log10(ys) if spec.get("logy") else ys, pen=pen, connect="finite", name=None if is_band else (names[j - 1] if j - 1 < len(names) else None)))
            if band:
                p.addItem(pg.FillBetweenItem(curves[band[0]], curves[band[1]], brush=pg.mkBrush(c["accent"] + "33")))
            for m in spec.get("markers", []):
                p.addItem(pg.InfiniteLine(pos=to_ts(m), angle=90, pen=pg.mkPen(c["text_dim"], width=1, style=Qt.PenStyle.DotLine)))
            self._labels(p, {**spec, "ylabel": spec.get("ylabel", "") + (" (log10)" if spec.get("logy") else "")})
            self.stack.setCurrentWidget(p)
            return
        if kind == "line":
            p = self.time_plot
            p.clear()
            p.setLogMode(x=False, y=False)
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
            p.getAxis("left").setTicks(None)
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
                cols = [c["warning"], c["info"], c["positive"]]
                grid = np.linspace(xs.min(), xs.max(), 200)
                for i, cv in enumerate(spec.get("curves", [])):
                    with np.errstate(all="ignore"):
                        gy = predict(cv["model"], cv["params"], grid)
                    ok = np.isfinite(gy)
                    p.plot(grid[ok], gy[ok], pen=pg.mkPen(cols[i % 3], width=2), name=cv["name"])
                for mk in spec.get("marks", []):
                    p.addItem(pg.ScatterPlotItem(x=[mk["x"]], y=[mk["y"]], size=11, symbol="d", pen=pg.mkPen(c["warning"]), brush=pg.mkBrush(c["warning"] + "CC")))
                    t = pg.TextItem(mk["name"], color=c["text"], anchor=(0, 1.1))
                    t.setPos(mk["x"], mk["y"])
                    p.addItem(t)
            elif kind == "hist":
                idx = np.array([int(r[0]) for r in rows])
                cnt = np.array([float(r[1]) for r in rows])
                w, mn = spec["w"], spec["mn"]
                p.addItem(pg.BarGraphItem(x=mn + (idx + 0.5) * w, height=cnt, width=w * 0.95, brush=c["accent"] + "AA", pen=None))
                for ln in spec.get("lines", []):
                    p.addItem(pg.InfiniteLine(pos=ln["x"], angle=90, pen=pg.mkPen(c["warning"], width=2, style=Qt.PenStyle.DashLine),
                                              label=ln.get("label", ""), labelOpts={"position": 0.92, "color": c["text"], "movable": False}))
                fit = spec.get("fit")
                if fit:
                    xs = np.linspace(mn, mn + (idx.max() + 1) * w, 300)
                    p.plot(xs, stats.dist_pdf(fit["dist"], fit["params"], xs) * spec["n"] * w, pen=pg.mkPen(c["warning"], width=2), name=fit["dist"])
            elif kind == "qq":
                th = np.array([float(r[0]) for r in rows])
                sm = np.array([float(r[1]) for r in rows])
                p.addItem(pg.ScatterPlotItem(x=th, y=sm, size=5, pen=None, brush=pg.mkBrush(c["accent"] + "CC")))
                q1, q3 = np.percentile(sm, [25, 75])
                t1, t3 = np.percentile(th, [25, 75])
                slope = (q3 - q1) / (t3 - t1) if t3 != t1 else 1.0
                p.plot(th, q1 + slope * (th - t1), pen=pg.mkPen(c["warning"], width=2))
            elif kind == "acf":
                y = np.array([np.nan if r[1] is None else float(r[1]) for r in rows])
                y = y[np.isfinite(y)]
                if len(y) < 8:
                    return self.show_message("Too few points for an autocorrelation plot.")
                k = np.arange(len(y))
                y = y - np.polyval(np.polyfit(k, y, 1), k)
                maxlag = min(spec.get("maxlag", 60), len(y) // 2)
                ac = stats.acf(y, maxlag)
                p.addItem(pg.BarGraphItem(x=np.arange(1, maxlag + 1), height=ac[1:], width=0.6, brush=c["accent"], pen=None))
                band = 1.96 / np.sqrt(len(y))
                for v in (band, -band):
                    p.addItem(pg.InfiniteLine(pos=v, angle=0, pen=pg.mkPen(c["warning"], style=Qt.PenStyle.DashLine)))
            elif kind == "box":
                rows = rows[:12]
                xs = np.arange(len(rows))
                lo, q1, med, q3, hi = (np.array([float(r[i]) for r in rows]) for i in (1, 2, 3, 4, 5))
                p.addItem(pg.BarGraphItem(x=xs, y0=q1, y1=q3, width=0.55, brush=c["accent"] + "AA", pen=pg.mkPen(c["accent"])))
                seg = lambda a, b: np.column_stack([np.repeat(xs, 2), np.column_stack([a, b]).ravel()])
                for pts, col in ((seg(lo, q1), c["text_dim"]), (seg(q3, hi), c["text_dim"])):
                    p.addItem(pg.PlotCurveItem(pts[:, 0], pts[:, 1], connect="pairs", pen=pg.mkPen(col, width=1)))
                mpts = np.column_stack([np.column_stack([xs - 0.275, xs + 0.275]).ravel(), np.repeat(med, 2)])
                p.addItem(pg.PlotCurveItem(mpts[:, 0], mpts[:, 1], connect="pairs", pen=pg.mkPen(c["warning"], width=2)))
                ax.setTicks([[(i, str(r[0])[:10]) for i, r in enumerate(rows)]])
            elif kind == "heat":
                labels = spec["labels"]
                L = len(labels)
                m = np.zeros((L, L))
                for i, j, v in rows:
                    m[int(i), int(j)] = float(v)
                img = pg.ImageItem(m)
                cmap = pg.ColorMap([0.0, 0.5, 1.0], [(47, 155, 255), (230, 230, 230), (255, 92, 92)])
                img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
                img.setLevels((-1, 1))
                p.addItem(img)
                ticks = [[(i + 0.5, str(n)[:7]) for i, n in enumerate(labels)]]
                ax.setTicks(ticks)
                p.getAxis("left").setTicks(ticks)
                p.setAspectLocked(False)
        self._labels(p, spec)
        self.stack.setCurrentWidget(p)

    def current_plot(self):
        w = self.stack.currentWidget()
        return w if isinstance(w, pg.PlotWidget) else None

    def export_image(self, path: str):
        """PNG (1600 px wide) or SVG. pyqtgraph's SVG exporter breaks on some Qt versions, so SVG falls back to Qt's own generator."""
        from pyqtgraph.exporters import ImageExporter, SVGExporter
        plot = self.current_plot()
        if plot is None:
            raise ValueError("Nothing to export: draw a chart first.")
        if path.lower().endswith(".svg"):
            try:
                SVGExporter(plot.plotItem).export(path)
                return
            except Exception:
                self._export_svg_qt(plot, path)
                return
        exp = ImageExporter(plot.plotItem)
        exp.parameters()["width"] = 1600
        exp.export(path)

    @staticmethod
    def _export_svg_qt(plot, path):
        from PySide6.QtCore import QRect, QSize
        from PySide6.QtGui import QPainter
        from PySide6.QtSvg import QSvgGenerator
        size = plot.size()
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QSize(size.width(), size.height()))
        gen.setViewBox(QRect(0, 0, size.width(), size.height()))
        gen.setTitle("UIE chart")
        painter = QPainter(gen)
        try:
            plot.render(painter)
        finally:
            painter.end()
