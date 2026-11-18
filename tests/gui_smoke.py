"""Manual GUI smoke test (needs PySide6 + pyqtgraph):  QT_QPA_PLATFORM=offscreen python tests/gui_smoke.py <dataset.csv> [outdir]"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from uie.ui.theme_manager import ThemeManager
from uie.ui.main_window import MainWindow

data, out = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else ".")
app = QApplication([])
themes = ThemeManager(Path(__file__).resolve().parents[1] / "themes"); themes.apply(themes.default_id)
w = MainWindow(themes); w.show()


def wait(cond, t=90):
    t0 = time.time()
    while not cond() and time.time() - t0 < t:
        app.processEvents(); time.sleep(0.02)
    assert cond(), "timeout"; app.processEvents()


w.open_dataset(data); wait(lambda: w.sheet is not None)
print("analysis:", len(w.sheet.facts), "findings;", sorted({f.type for f in w.sheet.facts if f.chart})[:12])
for f in [f for f in w.sheet.facts if f.type in ("distribution_fit", "seasonality", "pca", "group_test", "nonlinear_association")]:   # draw each new finding chart
    w.find_tbl.clearSelection()
    w.show_reviewed.setChecked(True)
    r = w._row_facts.index(f); w.find_tbl.selectRow(r); wait(lambda: True)
    time.sleep(0.4); app.processEvents()
    print("  finding chart drawn:", f.type, "ok" if w.chart.current_plot() is not None else "(message only)")
# model tab
w.mod_target.setCurrentIndex(w.mod_target.findData("score"))
for i in range(w.mod_feats.count()):
    if w.mod_feats.item(i).text() in ("grp", "x"): w.mod_feats.item(i).setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Checked)
w.run_model(); wait(lambda: w.mod_view.report is not None); print("regression:", [f.id for f in w.mod_view.report.facts])
w.mod_view.find_tbl.selectRow(0); app.processEvents(); print("  regression chart:", w.mod_view.chart.current_plot() is not None)
w.mod_view.report = None
w.mod_x.setCurrentIndex(w.mod_x.findData("x")); w.mod_y.setCurrentIndex(w.mod_y.findData("u")); w.mod_at.setText("1,2,9")
w.run_curvefit(); wait(lambda: w.mod_view.report is not None); print("curve fit:", w.mod_view.report.summary["best_model"])
w.mod_view.find_tbl.selectRow(0); app.processEvents(); print("  curve chart:", w.mod_view.chart.current_plot() is not None)
# charts tab: every kind, plus PNG/SVG export
pb = w.plot_builder
cases = {"hist": ("skewed", None, None, True), "scatter": ("x", "u", None, True), "line": ("day", "visits", None, False), "bar": (None, "score", "grp", False),
         "box": (None, "score", "grp", False), "heat": (None, None, None, False), "qq": ("skewed", None, None, True), "acf": ("day", "visits", None, False)}
for kind, (x, y, g, ov) in cases.items():
    pb.kind.setCurrentIndex(pb.kind.findData(kind))
    for combo, v in ((pb.x, x), (pb.y, y), (pb.g, g)): combo.setCurrentIndex(max(combo.findData(v), 0))
    pb.overlay.setChecked(ov); pb.logx.setChecked(kind == "hist"); pb.chart._last = None
    pb.draw(); wait(lambda: pb.chart._last is not None, 30)
    for ext in ("png", "svg"):
        path = out / f"chart_{kind}.{ext}"
        QFileDialog.getSaveFileName = staticmethod(lambda *a, path=path, **k: (str(path), ""))
        pb.export(ext)
        assert path.exists() and path.stat().st_size > 500, path
    print(f"  chart {kind}: drawn and exported ({(out / f'chart_{kind}.png').stat().st_size:,} bytes png)")
pb.kind.setCurrentIndex(pb.kind.findData("hist")); pb.x.setCurrentIndex(pb.x.findData("grp")); pb.chart._last = None; pb.draw(); time.sleep(0.5); app.processEvents()
print("bad input message:", pb.chart.msg.text()[:70])
wait(lambda: not w._workers, 30); w.close(); app.processEvents()
print("GUI SMOKE OK")
os._exit(0)
