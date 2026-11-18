"""Manual GUI smoke test for the QUANT tab:  QT_QPA_PLATFORM=offscreen python tests/gui_quant_smoke.py prices_long.csv option_chain.csv
prices_long.csv needs columns date, ticker, close (several tickers); option_chain.csv needs strike, type, mid, yrs."""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from uie.ui.theme_manager import ThemeManager
from uie.ui.main_window import MainWindow

prices, chain = sys.argv[1], sys.argv[2]
app = QApplication([])
themes = ThemeManager(Path(__file__).resolve().parents[1] / "themes"); themes.apply(themes.default_id)
w = MainWindow(themes); w.show()
q = w.quant


def wait(cond, t=180):
    t0 = time.time()
    while not cond() and time.time() - t0 < t:
        app.processEvents(); time.sleep(0.02)
    assert cond(), "timeout"; app.processEvents()


def check(name, view, expect_types=()):
    wait(lambda: view.report is not None)
    r = view.report
    drawn = []
    for i in range(view.find_tbl.rowCount()):
        view.find_tbl.selectRow(i); app.processEvents()
        if r.facts[i].id in r.charts:
            time.sleep(0.15); app.processEvents()
            drawn.append(view.chart.current_plot() is not None)
    got = {f.type for f in r.facts}
    miss = [t for t in expect_types if t not in got]
    print(f"{name:12s} findings={len(r.facts):2d} tables={list(r.tables)[:4]} charts_drawn={sum(drawn)}/{len(drawn)} {'MISSING ' + str(miss) if miss else 'ok'}")
    assert not miss and all(drawn), name
    view.report = None


w.open_dataset(prices); wait(lambda: w.sheet is not None)
q.time.setCurrentIndex(q.time.findData("date")); q.value.setCurrentIndex(q.value.findData("close")); q.entity.setCurrentIndex(q.entity.findData("ticker"))
q.ent_value.setCurrentText("AAA")
q.run_arima(); check("ARIMA", q.v_ts, ("q_arima_model", "q_arima_forecast"))
q.run_garch(); check("GARCH", q.v_ts, ("q_garch_model",))
q.entity.setCurrentIndex(q.entity.findData("ticker")); q.run_coint(); check("COINT", q.v_co, ("q_coint_none",))
q.run_risk(); check("VaR/ES", q.v_rk, ("q_var_headline",))
q.bt_train.setValue(300); q.bt_test.setValue(60); q.bt_boot.setValue(300)
q.run_backtest(); check("BACKTEST", q.v_bt, ("q_bt_perf", "q_bt_walkforward"))
q.run_portfolio(); check("PORTFOLIO", q.v_pf, ("q_pf_weights", "q_pf_oos"))
q.run_pricer(); wait(lambda: q.v_pr.report is not None); print("PRICER      ", q.v_pr.report.facts[0].metrics["message"][:90])
for m in ("binomial", "monte-carlo", "bond"):
    q.pr_model.setCurrentIndex(q.pr_model.findData(m)); q.v_pr.report = None; q.run_pricer(); assert q.v_pr.report is not None, m
    print("PRICER      ", m, "ok")
w.open_dataset(chain); wait(lambda: w.sheet is not None and w.sheet.dataset == Path(chain).name)
for combo, name in ((q.opt_strike, "strike"), (q.opt_type, "type"), (q.opt_price, "mid"), (q.opt_years, "yrs")):
    combo.setCurrentIndex(combo.findData(name))
q.opt_spot.setText("100"); q.opt_rate.setValue(0.03); q.opt_div.setValue(0.01)
q.run_options(); check("OPTION CHAIN", q.v_opt, ("q_opt_smile",))
wait(lambda: not w._workers, 60); w.close(); app.processEvents()
print("QUANT GUI SMOKE OK")
os._exit(0)
