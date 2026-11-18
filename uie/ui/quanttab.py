"""QUANT tab: ARIMA, GARCH, cointegration, VaR/ES, backtesting, portfolio optimisation, option-chain diagnostics and a pricing calculator."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)
from ..core.quant import backtest as bt
from ..core.quant import derivatives as deriv
from ..core.quant.arima import arima
from ..core.quant.cointegration import cointegration
from ..core.quant.garch import garch
from ..core.quant.portfolio import NAMES as PF_NAMES, portfolio
from ..core.quant.risk import METHODS as RISK_METHODS, risk
from ..core.quant.series import load_panel, load_series
from .tools import ReportView


def _row(*ws):
    h = QHBoxLayout()
    for w in ws:
        h.addWidget(w, 1 if isinstance(w, (QComboBox, QLineEdit)) else 0)
    return h


def _spin(lo, hi, val, step=1, dec=0, prefix="", suffix=""):
    dbl = bool(dec) or hi > 2_000_000_000 or isinstance(step, float)
    w = QDoubleSpinBox() if dbl else QSpinBox()
    w.setRange(lo, hi)
    if dbl:
        w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    w.setPrefix(prefix)
    w.setSuffix(suffix)
    return w


def _hint(text):
    """One short line; the full explanation is the tooltip (long hints used to squeeze the results area)."""
    first = text.split(". ")[0].rstrip(".")
    l = QLabel(first + (" (hover for details)" if first != text.rstrip(".") else ""))
    l.setObjectName("Dim")
    l.setToolTip(text)
    return l


class QuantTab(QWidget):
    def __init__(self, host, colors):
        super().__init__()
        self.host, self.colors, self.views = host, colors, []
        lay = QVBoxLayout(self)
        self.time, self.value, self.entity = QComboBox(), QComboBox(), QComboBox()
        self.ent_value = QComboBox(); self.ent_value.setEditable(True); self.ent_value.setMinimumWidth(110)
        self.kind = QComboBox(); self.kind.addItem("Value is: price", "price"); self.kind.addItem("Value is: simple returns", "returns")
        self.unit = QComboBox()
        for u in ("day", "week", "month"):
            self.unit.addItem(f"Bucket: {u}", u)
        self.assets = QListWidget(); self.assets.setMaximumHeight(64)
        self.entity.currentIndexChanged.connect(self._fill_entity_values)
        lay.addLayout(_row(self.time, self.value, self.entity, self.ent_value, self.kind, self.unit))
        lay.addLayout(_row(QLabel("Wide-table asset columns (optional, for COINT / PORTFOLIO):"), self.assets))
        lay.addWidget(_hint("Series: pick the time column, the price (or return) column and, for a long table, the entity column and which entity. COINT and PORTFOLIO use all entities of a long table, "
                            "or the ticked columns of a wide table. Results are model-based descriptions with their assumptions stated; they are not investment advice."))
        self.inner = QTabWidget()
        lay.addWidget(self.inner, 1)
        self._build_series_models(); self._build_coint(); self._build_risk(); self._build_backtest(); self._build_portfolio(); self._build_options(); self._build_pricer()

    # ---- plumbing ----
    def _view(self, name):
        v = ReportView(self.host.status, self.colors)
        self.views.append(v)
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(v)
        self.inner.addTab(w, name)
        return v

    def set_colors(self, colors):
        self.colors = colors
        for v in self.views:
            v.set_colors(colors)

    def set_columns(self, schema, temporal, sem):
        names = [n for n, _ in schema]
        for combo, head in ((self.time, "Time"), (self.value, "Value"), (self.entity, "Entity")):
            keep = combo.currentData(); combo.blockSignals(True); combo.clear()
            combo.addItem(f"{head}: " + ("none" if head == "Entity" else "choose"), None)
            for n in names:
                combo.addItem(f"{head}: {n}", n)
            combo.blockSignals(False)
            combo.setCurrentIndex(max(combo.findData(keep), 0))
        if self.time.currentData() is None and temporal:
            self.time.setCurrentIndex(self.time.findData(temporal[0]))
        if self.value.currentData() is None:
            pr = [n for n in sem.cols_with_role("price") if n in names] or [n for n in names if n.lower() in ("close", "adj_close", "price", "nav")]
            if pr:
                self.value.setCurrentIndex(self.value.findData(pr[0]))
        if self.entity.currentData() is None and sem.get("entity") in names:
            self.entity.setCurrentIndex(self.entity.findData(sem.get("entity")))
        self.assets.clear()
        for n, t in schema:
            if str(t).upper() in ("DOUBLE", "FLOAT", "REAL") or str(t).upper().startswith("DECIMAL"):
                it = QListWidgetItem(n); it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable); it.setCheckState(Qt.CheckState.Unchecked); self.assets.addItem(it)
        for combo, key in ((self.opt_strike, None), (self.opt_type, None), (self.opt_price, None), (self.opt_expiry, None), (self.opt_years, None), (self.opt_bid, None), (self.opt_ask, None), (self.bt_signal, None)):
            keep = combo.currentData(); combo.clear(); combo.addItem("(none)", None)
            for n in names:
                combo.addItem(n, n)
            combo.setCurrentIndex(max(combo.findData(keep), 0))
        self._fill_entity_values()

    def _fill_entity_values(self):
        eng = self.host.engine
        self.ent_value.clear()
        e = self.entity.currentData()
        if e and eng.src:
            try:
                _, rows = eng.evidence(f'SELECT CAST("{e}" AS VARCHAR) FROM data WHERE "{e}" IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC LIMIT 60')
                for r in rows:
                    self.ent_value.addItem(r[0])
            except Exception:
                pass

    def _ready(self):
        if not self.host.engine.src:
            self.host.status("OPEN A DATASET FIRST")
            return False
        return True

    def _series_job(self):
        eng = self.host.engine
        src, sem = eng.src, eng.sem
        t, v, e, ev = self.time.currentData(), self.value.currentData(), self.entity.currentData(), self.ent_value.currentText().strip() or None
        kind, unit = self.kind.currentData(), self.unit.currentData()
        return lambda p: load_series(src.path, t, v, e, ev, src.table, sem if (sem.columns or sem.dataset) else None, unit, kind, progress=p)

    def _panel_job(self):
        eng = self.host.engine
        src, sem = eng.src, eng.sem
        cols = [self.assets.item(i).text() for i in range(self.assets.count()) if self.assets.item(i).checkState() == Qt.CheckState.Checked]
        t, v, e = self.time.currentData(), self.value.currentData(), self.entity.currentData()
        kind, unit = self.kind.currentData(), self.unit.currentData()
        return lambda p: load_panel(src.path, t, v, e, cols or None, src.table, sem if (sem.columns or sem.dataset) else None, unit, kind, progress=p)

    def _go(self, job, view):
        self.host._busy(True)
        def done(r):
            self.host._busy(False)
            view.show_report(r)
            self.host.status(f"{r.title}: {len(r.facts)} findings")
        self.host._run(job, done)

    # ---- tabs ----
    def _build_series_models(self):
        v = self._view("ARIMA / GARCH")
        self.ts_horizon = _spin(1, 250, 20, prefix="horizon "); self.ts_maxp = _spin(0, 4, 2, prefix="max p "); self.ts_maxq = _spin(0, 4, 2, prefix="max q ")
        self.ts_log = QCheckBox("Log prices")
        a = QPushButton("RUN ARIMA"); a.setProperty("accent", True); a.clicked.connect(self.run_arima)
        g = QPushButton("RUN GARCH"); g.setProperty("accent", True); g.clicked.connect(self.run_garch)
        v.controls.addLayout(_row(self.ts_horizon, self.ts_maxp, self.ts_maxq, self.ts_log, a, g))
        v.controls.addWidget(_hint("ARIMA picks (p,d,q) by AIC with d from unit-root tests, forecasts with prediction intervals and tests whether it beats a random walk out of sample. "
                                   "GARCH / GJR-GARCH (normal or Student-t) models volatility clustering, leverage and fat tails and tests its volatility forecast against EWMA."))
        self.v_ts = v

    def run_arima(self):
        if not self._ready(): return
        job, h, mp, mq, lg, cfg = self._series_job(), self.ts_horizon.value(), self.ts_maxp.value(), self.ts_maxq.value(), self.ts_log.isChecked(), self.host.engine.cfg
        self._go(lambda p: arima(job(p), None, mp, mq, h, lg, cfg, p, "UIE app"), self.v_ts)

    def run_garch(self):
        if not self._ready(): return
        job, h, cfg = self._series_job(), self.ts_horizon.value(), self.host.engine.cfg
        self._go(lambda p: garch(job(p), horizon=h, cfg=cfg, progress=p, command="UIE app"), self.v_ts)

    def _build_coint(self):
        v = self._view("COINTEGRATION")
        self.co_log = QCheckBox("Log prices"); self.co_log.setChecked(True)
        b = QPushButton("SCAN PAIRS"); b.setProperty("accent", True); b.clicked.connect(self.run_coint)
        v.controls.addLayout(_row(self.co_log, b))
        v.controls.addWidget(_hint("Engle-Granger test on every pair (both orderings, MacKinnon p-values, Benjamini-Hochberg corrected), with hedge ratio, spread half-life, current spread z-score, "
                                   "hedge-ratio stability across the two halves, and a spurious-regression check."))
        self.v_co = v

    def run_coint(self):
        if not self._ready(): return
        job, lg, cfg = self._panel_job(), self.co_log.isChecked(), self.host.engine.cfg
        self._go(lambda p: cointegration(job(p), lg, cfg=cfg, progress=p, command="UIE app"), self.v_co)

    def _build_risk(self):
        v = self._view("VaR / ES")
        self.rk_conf = {c: QCheckBox(f"{c:.0%}") for c in (0.9, 0.95, 0.975, 0.99)}
        for c, w in self.rk_conf.items(): w.setChecked(c in (0.95, 0.99))
        self.rk_methods = {m: QCheckBox(n) for m, n in RISK_METHODS.items()}
        for m, w in self.rk_methods.items(): w.setChecked(m != "garch")
        self.rk_window = _spin(120, 5000, 500, 10, prefix="window "); self.rk_horizon = _spin(1, 60, 1, prefix="horizon "); self.rk_pos = _spin(0, 1e12, 0, 1000, 0, "position ")
        b = QPushButton("RUN VaR / ES"); b.setProperty("accent", True); b.clicked.connect(self.run_risk)
        v.controls.addLayout(_row(*self.rk_conf.values(), self.rk_window, self.rk_horizon, self.rk_pos))
        v.controls.addLayout(_row(*self.rk_methods.values(), b))
        v.controls.addWidget(_hint("Rolling out-of-sample backtests: Kupiec (coverage), Christoffersen (independence), Basel traffic light, and the ratio of realised losses to expected shortfall. "
                                   "GARCH-FHS refits the model periodically and is slow."))
        self.v_rk = v

    def run_risk(self):
        if not self._ready(): return
        conf = tuple(c for c, w in self.rk_conf.items() if w.isChecked()) or (0.95, 0.99)
        ms = tuple(m for m, w in self.rk_methods.items() if w.isChecked()) or ("historical",)
        job, win, hz, pos, cfg = self._series_job(), self.rk_window.value(), self.rk_horizon.value(), self.rk_pos.value() or None, self.host.engine.cfg
        self._go(lambda p: risk(job(p), conf, ms, win, hz, pos, cfg=cfg, progress=p, command="UIE app"), self.v_rk)

    def _build_backtest(self):
        v = self._view("BACKTEST")
        self.bt_strat = QComboBox()
        for k in ("sma", "momentum", "meanrev", "rsi", "breakout", "buyhold"):
            self.bt_strat.addItem(bt.LABELS[k], k)
        self.bt_params = QLineEdit(); self.bt_params.setPlaceholderText("parameters, e.g. fast=20,slow=100 (blank = defaults)")
        self.bt_signal = QComboBox()
        self.bt_ls = QCheckBox("Long/short"); self.bt_anch = QCheckBox("Anchored"); self.bt_walk = QCheckBox("Walk-forward"); self.bt_walk.setChecked(True)
        self.bt_cost = _spin(0, 500, 5, 1, 1, "cost ", " bps"); self.bt_slip = _spin(0, 500, 0, 1, 1, "slippage ", " bps"); self.bt_vt = _spin(0, 2, 0, 0.05, 2, "vol target ")
        self.bt_train = _spin(100, 10000, 750, 25, prefix="train "); self.bt_test = _spin(20, 2000, 125, 5, prefix="test "); self.bt_boot = _spin(200, 10000, 2000, 200, prefix="bootstrap ")
        b = QPushButton("RUN BACKTEST"); b.setProperty("accent", True); b.clicked.connect(self.run_backtest)
        v.controls.addLayout(_row(self.bt_strat, self.bt_params, QLabel("or signal column:"), self.bt_signal))
        v.controls.addLayout(_row(self.bt_ls, self.bt_walk, self.bt_anch, self.bt_cost, self.bt_slip, self.bt_vt))
        v.controls.addLayout(_row(self.bt_train, self.bt_test, self.bt_boot, b))
        v.controls.addWidget(_hint("A position decided at the close of a bar earns the next bar's return (no look-ahead). Walk-forward re-optimises the parameter grid on each training window and trades the "
                                   "next test window; the report adds drawdown episodes, walk-forward efficiency, deflated Sharpe, bootstrap robustness and cost sensitivity."))
        self.v_bt = v

    def run_backtest(self):
        if not self._ready(): return
        job, strat = self._series_job(), self.bt_strat.currentData()
        sigc, cfg = self.bt_signal.currentData(), self.host.engine.cfg
        params = {}
        for part in [x.strip() for x in self.bt_params.text().split(",") if x.strip()]:
            k, _, val = part.partition("=")
            try:
                f = float(val); params[k] = int(f) if f == int(f) else f
            except ValueError:
                return self.host.status(f"BAD PARAMETER '{part}': use name=number")
        o = dict(long_short=self.bt_ls.isChecked(), cost_bps=self.bt_cost.value(), slippage_bps=self.bt_slip.value(), vol_target=self.bt_vt.value() or None, train=self.bt_train.value(),
                 test=self.bt_test.value(), anchored=self.bt_anch.isChecked(), walk=self.bt_walk.isChecked(), boot=self.bt_boot.value())
        eng = self.host.engine
        src, t, e, ev, unit = eng.src, self.time.currentData(), self.entity.currentData(), self.ent_value.currentText().strip() or None, self.unit.currentData()

        def run(p):
            s = job(p)
            signal = None
            if sigc:
                sg = load_series(src.path, t, sigc, e, ev, src.table, None, unit, "returns")
                m = dict(zip(sg.dates, sg.values)); signal = [m.get(d, 0.0) for d in s.dates]
            return bt.backtest(s, "signal" if sigc else strat, params or None, signal=signal, cfg=cfg, progress=p, command="UIE app", **o)
        self._go(run, self.v_bt)

    def _build_portfolio(self):
        v = self._view("PORTFOLIO")
        self.pf_methods = {m: QCheckBox(n) for m, n in PF_NAMES.items()}
        for w in self.pf_methods.values(): w.setChecked(True)
        self.pf_rf = _spin(0, 0.5, 0, 0.005, 3, "risk-free "); self.pf_min = _spin(-5, 1, 0, 0.05, 2, "min weight "); self.pf_max = _spin(0.01, 5, 1, 0.05, 2, "max weight ")
        self.pf_look = _spin(60, 2000, 252, 10, prefix="lookback "); self.pf_reb = _spin(1, 252, 21, 1, prefix="rebalance "); self.pf_cost = _spin(0, 500, 5, 1, 1, "cost ", " bps")
        b = QPushButton("OPTIMISE"); b.setProperty("accent", True); b.clicked.connect(self.run_portfolio)
        v.controls.addLayout(_row(*self.pf_methods.values()))
        v.controls.addLayout(_row(self.pf_rf, self.pf_min, self.pf_max, self.pf_look, self.pf_reb, self.pf_cost, b))
        v.controls.addWidget(_hint("Ledoit-Wolf shrunk covariance; weights bounded and summing to 1 (long-only by default). The out-of-sample test re-optimises on a rolling window and reports whether any "
                                   "optimised portfolio beats equal weight with statistical confidence."))
        self.v_pf = v

    def run_portfolio(self):
        if not self._ready(): return
        ms = tuple(m for m, w in self.pf_methods.items() if w.isChecked()) or ("equal",)
        job, cfg = self._panel_job(), self.host.engine.cfg
        a = (self.pf_rf.value(), self.pf_min.value(), self.pf_max.value(), self.pf_look.value(), self.pf_reb.value(), self.pf_cost.value())
        self._go(lambda p: portfolio(job(p), ms, *a, cfg=cfg, progress=p, command="UIE app"), self.v_pf)

    def _build_options(self):
        v = self._view("OPTION CHAIN")
        self.opt_strike, self.opt_type, self.opt_price, self.opt_expiry, self.opt_years, self.opt_bid, self.opt_ask = (QComboBox() for _ in range(7))
        self.opt_spot = QLineEdit(); self.opt_spot.setPlaceholderText("spot: number or column")
        self.opt_rate = _spin(-0.1, 1, 0.03, 0.005, 4, "rate "); self.opt_div = _spin(0, 1, 0, 0.005, 4, "dividend yield "); self.opt_val = QLineEdit(); self.opt_val.setPlaceholderText("valuation date YYYY-MM-DD")
        b = QPushButton("ANALYSE CHAIN"); b.setProperty("accent", True); b.clicked.connect(self.run_options)
        v.controls.addLayout(_row(QLabel("strike"), self.opt_strike, QLabel("call/put"), self.opt_type, QLabel("price"), self.opt_price, QLabel("bid"), self.opt_bid, QLabel("ask"), self.opt_ask))
        v.controls.addLayout(_row(QLabel("expiry date"), self.opt_expiry, QLabel("or years"), self.opt_years, self.opt_spot, self.opt_val))
        v.controls.addLayout(_row(self.opt_rate, self.opt_div, b))
        v.controls.addWidget(_hint("Implied volatility per quote, then no-arbitrage checks (prices outside bounds, monotonicity, butterfly convexity, put-call parity), the smile and the term structure. "
                                   "Quotes within a price tick of intrinsic value carry no volatility information and are not flagged."))
        self.v_opt = v

    def run_options(self):
        if not self._ready(): return
        g = lambda c: c.currentData()
        spot = self.opt_spot.text().strip()
        try: spot = float(spot)
        except ValueError: pass
        if not (g(self.opt_strike) and g(self.opt_type) and spot not in ("", None) and (g(self.opt_price) or (g(self.opt_bid) and g(self.opt_ask))) and (g(self.opt_expiry) or g(self.opt_years))):
            return self.host.status("CHOOSE STRIKE, TYPE, PRICE (or BID+ASK), EXPIRY (or YEARS) AND ENTER THE SPOT")
        eng = self.host.engine; src, sem, cfg = eng.src, eng.sem, eng.cfg
        a = (src.path, g(self.opt_strike), g(self.opt_type), g(self.opt_price), g(self.opt_bid), g(self.opt_ask), g(self.opt_expiry), g(self.opt_years), spot, self.opt_rate.value(), self.opt_div.value(),
             self.opt_val.text().strip() or None, src.table, sem if (sem.columns or sem.dataset) else None, cfg)
        self._go(lambda p: deriv.option_chain(*a, progress=p, command="UIE app"), self.v_opt)

    def _build_pricer(self):
        v = self._view("PRICER")
        self.pr_model = QComboBox()
        for lbl, k in (("Black-Scholes", "black-scholes"), ("Binomial (CRR)", "binomial"), ("Monte Carlo", "monte-carlo"), ("Bond", "bond")):
            self.pr_model.addItem(lbl, k)
        self.pr_kind = QComboBox(); self.pr_kind.addItems(["call", "put"]); self.pr_style = QComboBox(); self.pr_style.addItems(["european", "asian", "down_out", "up_out"])
        self.pr_am = QCheckBox("American")
        self.pr_S = _spin(0.01, 1e9, 100, 1, 4, "S "); self.pr_K = _spin(0.01, 1e9, 100, 1, 4, "K "); self.pr_T = _spin(0.001, 100, 1, 0.25, 4, "T (years) ")
        self.pr_r = _spin(-0.1, 1, 0.05, 0.005, 4, "r "); self.pr_q = _spin(0, 1, 0.0, 0.005, 4, "q "); self.pr_sig = _spin(0.001, 5, 0.2, 0.01, 4, "vol "); self.pr_bar = _spin(0, 1e9, 0, 1, 2, "barrier ")
        self.bd_face = _spin(1, 1e9, 100, 10, 2, "face "); self.bd_cpn = _spin(0, 1, 0.05, 0.005, 4, "coupon "); self.bd_yrs = _spin(0.25, 100, 10, 1, 2, "years "); self.bd_freq = _spin(1, 12, 2, 1, prefix="per year ")
        self.bd_ytm = _spin(-0.5, 5, 0.05, 0.005, 4, "yield "); self.bd_price = _spin(0, 1e9, 0, 1, 4, "or price (0 = use yield) ")
        b = QPushButton("PRICE"); b.setProperty("accent", True); b.clicked.connect(self.run_pricer)
        v.controls.addLayout(_row(self.pr_model, self.pr_kind, self.pr_style, self.pr_am, self.pr_S, self.pr_K, self.pr_T))
        v.controls.addLayout(_row(self.pr_r, self.pr_q, self.pr_sig, self.pr_bar))
        v.controls.addLayout(_row(self.bd_face, self.bd_cpn, self.bd_yrs, self.bd_freq, self.bd_ytm, self.bd_price, b))
        v.controls.addWidget(_hint("Does not need a dataset. Black-Scholes-Merton with Greeks, a CRR tree (American exercise), Monte Carlo with antithetic variates (European, Asian, barrier) and a bond calculator "
                                   "(price, yield, duration, convexity, DV01). Vega is per vol point, theta per day, rho per 1% rate."))
        self.v_pr = v

    def run_pricer(self):
        m = self.pr_model.currentData()
        try:
            if m == "bond":
                r = deriv.price_bond(self.bd_face.value(), self.bd_cpn.value(), self.bd_yrs.value(), self.bd_freq.value(), None if self.bd_price.value() else self.bd_ytm.value(), self.bd_price.value() or None)
            else:
                r = deriv.price_option(m, self.pr_S.value(), self.pr_K.value(), self.pr_T.value(), self.pr_r.value(), self.pr_q.value(), self.pr_sig.value(), self.pr_kind.currentText(),
                                       self.pr_am.isChecked(), self.pr_style.currentText(), self.pr_bar.value() or None)
        except ValueError as e:
            return self.host.status("PRICER: " + str(e))
        self.v_pr.show_report(r)
        self.host.status(f"{r.title}: done")
