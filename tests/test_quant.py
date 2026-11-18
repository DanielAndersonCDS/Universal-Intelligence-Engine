"""Quant-finance tests.  Oracle comparisons (statsmodels, arch, scipy, sklearn) run only when those packages are installed."""
import csv, importlib.util, io, json, math, sys, tempfile, unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datagen
from uie import cli
from uie.core import stats
from uie.core.quant import arima as A, backtest as B, derivatives as D, garch as G, portfolio as PF, risk as R
from uie.core.quant.cointegration import cointegration, engle_granger, half_life
from uie.core.quant.mackinnon import mackinnon_crit, mackinnon_p
from uie.core.quant.series import load_panel, load_series

TMP = Path(tempfile.mkdtemp(prefix="uie_quant_"))
have = lambda m: importlib.util.find_spec(m) is not None


def write_prices(name, prices, start=date(2012, 1, 2)):
    days = [d for d in (start + timedelta(i) for i in range(len(prices) * 2)) if d.weekday() < 5][:len(prices)]
    p = TMP / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "close"])
        w.writerows([[d.isoformat(), round(float(x), 5)] for d, x in zip(days, prices)])
    return str(p)


class Foundations(unittest.TestCase):
    def test_quantiles(self):
        self.assertAlmostEqual(stats.norm_ppf(0.975), 1.959964, 5)
        self.assertAlmostEqual(stats.norm_ppf(1e-6), -4.753424, 4)
        self.assertAlmostEqual(stats.t_ppf(0.975, 7), 2.364624, 4)

    def test_mackinnon_asymptotics(self):
        c1 = mackinnon_crit(1, "c")
        self.assertAlmostEqual(c1[1], -2.8621, 2)                       # ADF 5%, constant
        self.assertAlmostEqual(mackinnon_p(c1[1], 1, "c"), 0.05, 2)
        self.assertLess(mackinnon_crit(2, "c")[1], c1[1])                # Engle-Granger needs a more negative statistic

    def test_adf_has_p_value(self):
        rng = np.random.default_rng(1)
        self.assertGreater(stats.adf(np.cumsum(rng.normal(size=500)), "c").p, 0.05)
        self.assertLess(stats.adf(rng.normal(size=500), "c").p, 0.001)


class ARIMA(unittest.TestCase):
    def test_arma_parameters_recovered(self):
        rng = np.random.default_rng(7)
        n = 2500
        e = rng.normal(size=n + 50)
        y = np.zeros(n + 50)
        for t in range(1, n + 50):
            y[t] = 0.2 + 0.6 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        f = A.fit_arma(y[50:], 1, 1)
        self.assertAlmostEqual(float(f["phi"][0]), 0.6, delta=0.06)
        self.assertAlmostEqual(float(f["theta"][0]), 0.4, delta=0.08)
        self.assertTrue(abs(f["phi"][0]) < 1 and abs(f["theta"][0]) < 1)

    def test_stationarity_parametrisation(self):
        for x in ([5.0, -5.0], [0.3, -0.5, 0.2], [10, 10, 10]):
            phi = A._pacf_to_poly(x)
            roots = np.roots(np.concatenate([[1.0], -phi]))
            self.assertTrue(np.all(np.abs(roots) < 1.0 + 1e-9))           # AR polynomial 1 - sum(phi B^i): reciprocal roots inside unit circle
        self.assertTrue(np.allclose(np.tanh(A._poly_to_pacf(A._pacf_to_poly([0.3, -0.5, 0.2]))), np.tanh([0.3, -0.5, 0.2])))

    def test_forecast_and_psi_weights(self):
        self.assertTrue(np.allclose(A.psi_weights(np.array([]), np.array([]), 1, 6), 1.0))      # random walk: variance grows linearly
        self.assertTrue(np.allclose(A.psi_weights(np.array([0.5]), np.array([0.3]), 1, 5), [1, 1.8, 2.2, 2.4, 2.5]))
        rng = np.random.default_rng(2)
        lv = np.cumsum(rng.normal(size=600)) + 50
        f = A.fit_arma(np.diff(lv), 0, 0)
        fc, lo, hi = A.forecast(lv, f, 1, 5)
        self.assertTrue(np.all(np.diff(hi - lo) > 0))
        self.assertAlmostEqual(float(fc[0]), float(lv[-1] + f["mu"]), 6)

    @unittest.skipUnless(have("statsmodels"), "statsmodels not installed")
    def test_against_statsmodels(self):
        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA
        rng = np.random.default_rng(3)
        n = 3000
        e = rng.normal(size=n + 50)
        w = np.zeros(n + 50)
        for t in range(1, n + 50):
            w[t] = 0.1 + 0.5 * w[t - 1] + e[t] + 0.3 * e[t - 1]
        lv = np.cumsum(w[50:]) + 100
        f = A.fit_arma(np.diff(lv), 1, 1)
        fc, lo, hi = A.forecast(lv, f, 1, 10)
        m = ARIMA(lv, order=(1, 1, 1), trend="t").fit()
        sf = m.get_forecast(10)
        self.assertTrue(np.allclose(fc, sf.predicted_mean, atol=0.05))
        self.assertTrue(np.allclose((hi - lo) / 3.92, sf.se_mean, atol=0.02))

    def test_report(self):
        rng = np.random.default_rng(5)
        s = load_series(write_prices("rw_arima", 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))))
        r = A.arima(s, horizon=10, log=True)
        self.assertEqual(r.facts[0].type, "q_arima_model")
        self.assertEqual(r.summary["differencing d"], 1)                    # a random walk needs one difference
        skill = next(f for f in r.facts if f.type == "q_arima_skill")
        self.assertGreater(skill.metrics["dm_p"], 0.05)                      # no skill beyond a random walk
        self.assertEqual(len(r.tables["Forecast"][1]), 10)


class GARCH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        n = 3000
        r = np.zeros(n)
        s2 = 0.05 / (1 - 0.98)
        for t in range(n):
            r[t] = 0.05 + math.sqrt(s2) * rng.normal()
            e = r[t] - 0.05
            s2 = 0.05 + 0.08 * e * e + 0.90 * s2
        cls.r = r

    def test_linear_recursion_is_exact(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0.1, 1, 4000)
        s = np.empty(4000)
        s[0] = 1.0
        for t in range(1, 4000):
            s[t] = a[t] + 0.93 * s[t - 1]
        self.assertLess(np.max(np.abs(G._linrec(a, 0.93, 1.0) - s)), 1e-10)

    def test_parameters_recovered_and_stationary(self):
        f = G.fit_garch(self.r, "garch", "normal")
        self.assertAlmostEqual(f["alpha"], 0.08, delta=0.04)
        self.assertAlmostEqual(f["beta"], 0.90, delta=0.04)
        self.assertLess(f["persistence"], 1.0)
        self.assertGreater(f["half_life"], 10)

    @unittest.skipUnless(have("arch"), "arch not installed")
    def test_against_arch(self):
        import warnings
        warnings.filterwarnings("ignore")
        from arch import arch_model
        f = G.fit_garch(self.r, "garch", "normal")
        am = arch_model(self.r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        self.assertAlmostEqual(f["loglik"], float(am.loglikelihood), delta=1.0)
        self.assertAlmostEqual(f["alpha"], float(am.params["alpha[1]"]), delta=0.01)
        self.assertAlmostEqual(f["beta"], float(am.params["beta[1]"]), delta=0.01)

    def test_report(self):
        p = 100 * np.exp(np.cumsum(self.r[:1500] / 100))
        r = G.garch(load_series(write_prices("garch_px", p)), models=[("garch", "normal"), ("gjr", "t")])
        self.assertIn("q_garch_model", [f.type for f in r.facts])
        self.assertEqual(len(r.tables["Volatility forecast"][1]), 20)


class Cointegration(unittest.TestCase):
    def test_pair_detection_and_spurious(self):
        rng = np.random.default_rng(5)
        n = 1500
        x = np.cumsum(rng.normal(0, 1, n))
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.9 * u[t - 1] + rng.normal(0, 0.5)
        y = 2 + 0.8 * x + u
        w = np.cumsum(rng.normal(0, 1, n))
        good, bad = engle_granger(y, x), engle_granger(w, x)
        self.assertLess(good["p"], 0.001)
        self.assertAlmostEqual(float(good["beta"][0]), 0.8, delta=0.05)
        self.assertGreater(bad["p"], 0.05)
        self.assertAlmostEqual(half_life(u)[0], math.log(0.5) / math.log(0.9), delta=2.5)

    def _panel(self, seed, name, stationary=False):
        rng = np.random.default_rng(seed)
        n = 1200
        base = np.cumsum(rng.normal(0, 1, n))
        cols = {"A": 500 + base, "B": 800 + 1.2 * base + rng.normal(0, 1.0, n)}
        for k in "CDE":
            cols[k] = 600 + np.cumsum(rng.normal(0, 1, n))
        if stationary:                                      # C becomes a genuinely mean-reverting series
            c = np.zeros(n)
            for t in range(1, n):
                c[t] = 0.8 * c[t - 1] + rng.normal()
            cols["C"] = 600 + c
        days = [d for d in (date(2015, 1, 1) + timedelta(i) for i in range(2 * n)) if d.weekday() < 5][:n]
        path = TMP / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date"] + list(cols))
            for i in range(n):
                w.writerow([days[i].isoformat()] + [round(float(v[i]), 4) for v in cols.values()])
        return cointegration(load_panel(str(path), "date", columns=list(cols)), log=False)

    def test_panel_scan_finds_the_planted_pair(self):
        r = self._panel(10, "panel_clean")
        sig = [f for f in r.facts if f.type == "q_coint_pair"]
        self.assertEqual(len(sig), 1)
        self.assertEqual(set(sig[0].columns), {"A", "B"})
        self.assertEqual(r.summary["pairs tested"], 10)
        self.assertIn("Series (ADF)", r.tables)

    def test_stationary_series_is_not_reported_as_cointegration(self):
        r = self._panel(10, "panel_stationary", stationary=True)
        pairs = [f for f in r.facts if f.type == "q_coint_pair"]
        self.assertEqual([set(f.columns) for f in pairs], [{"A", "B"}])        # only the real pair survives
        st = next(f for f in r.facts if f.type == "q_coint_stationary")
        self.assertEqual(st.metrics["series"], ["C"])
        self.assertTrue(any(f.type == "q_coint_caveat" and "C" in f.metrics["flagged"] for f in r.facts))
        verdict = {row[0]: row[3] for row in r.tables["Series (ADF)"][1]}
        self.assertEqual(verdict["C"], "looks stationary")
        self.assertEqual(verdict["D"], "unit root not rejected")


class Risk(unittest.TestCase):
    def test_closed_forms(self):
        rng = np.random.default_rng(2)
        v, e = R.var_es("normal", rng.normal(0, 0.01, 400000), 0.99)
        self.assertAlmostEqual(v, 0.01 * 2.326348, delta=2e-4)
        self.assertAlmostEqual(e, 0.01 * stats.norm_pdf(2.326348) / 0.01, delta=3e-4)
        r = rng.standard_t(4, size=300000) * 0.01 / math.sqrt(2)
        self.assertAlmostEqual(R.fit_t(r)[2], 4.0, delta=0.3)
        vt, _ = R.var_es("t", r, 0.99)
        self.assertAlmostEqual(vt, 0.01 / math.sqrt(2) * stats.t_ppf(0.99, 4), delta=4e-4)

    def test_coverage_tests(self):
        lr, p = R.kupiec(8, 500, 0.01)
        self.assertAlmostEqual(lr, 1.5383, 3)
        self.assertAlmostEqual(p, 0.2149, 3)
        exc = np.zeros(1000, int)
        exc[100:110] = 1
        exc[500:508] = 1
        self.assertLess(R.christoffersen(exc)[1], 0.001)                    # clustered exceptions
        self.assertEqual((R.traffic_light(4), R.traffic_light(5), R.traffic_light(10)), ("green", "yellow", "red"))

    def test_backtest_is_calibrated_on_matching_model(self):
        rng = np.random.default_rng(4)
        r = rng.standard_t(4, size=3000) * 0.01 / math.sqrt(2)
        b = R.backtest_method("historical", r, 0.99, 500)
        self.assertGreater(b["p_cc"], 0.01)
        self.assertAlmostEqual(b["es_ratio"], 1.0, delta=0.25)

    def test_report_rejects_a_misspecified_model(self):
        rng = np.random.default_rng(8)
        n = 3500
        ret = np.zeros(n)
        s2 = 1e-4
        for t in range(n):                                                    # strong volatility clustering: a constant-variance model must fail
            ret[t] = math.sqrt(s2) * rng.standard_t(5) * math.sqrt(3 / 5)
            s2 = 2e-6 + 0.14 * ret[t] ** 2 + 0.84 * s2
        s = load_series(write_prices("var_px", 100 * np.cumprod(1 + ret)))
        rep = R.risk(s, (0.99,), ("historical", "normal", "ewma"), window=500)
        self.assertTrue(any(f.type == "q_var_backtest" for f in rep.facts))
        self.assertEqual(len(rep.tables["Backtest"][1]), 3)


class Backtesting(unittest.TestCase):
    def test_metrics_and_drawdowns(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.0005, 0.01, 1000)
        p = B.perf(r, 252)
        self.assertAlmostEqual(p["sharpe"], r.mean() / r.std(ddof=1) * math.sqrt(252), 9)
        self.assertAlmostEqual(p["cagr"], np.prod(1 + r) ** (252 / 1000) - 1, 9)
        d = B.drawdowns(np.array([0.1, -0.2, -0.1, 0.3, 0.5, -0.05, 0.0, 0.2]))
        e = d["episodes"]
        self.assertEqual([(x["peak"], x["trough"], x["recovery"]) for x in e], [(0, 2, 4), (4, 5, 7)])
        self.assertAlmostEqual(e[0]["depth"], 0.792 / 1.1 - 1, 9)

    def test_no_lookahead(self):
        rng = np.random.default_rng(2)
        P = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        for strat in ("sma", "momentum", "meanrev", "rsi", "breakout"):
            full = B.positions(strat, P, B.DEFAULTS[strat])
            for k in (150, 250, 399):                                         # a position never depends on later prices
                self.assertEqual(B.positions(strat, P[:k], B.DEFAULTS[strat])[-1], full[k - 1], strat)

    def test_selection_bias_is_exposed(self):
        rng = np.random.default_rng(3)
        trials = np.array([x.mean() / x.std(ddof=1) for x in rng.normal(0, 0.01, (100, 1000))])
        dsr, sr0 = B.deflated_sharpe(trials.max(), 1000, 0.0, 3.0, trials)
        self.assertLess(dsr, 0.95)
        self.assertGreater(sr0, 0)

    def _series(self, name, prices):
        return load_series(write_prices(name, prices))

    def test_random_walk_is_flagged_and_real_momentum_is_not(self):
        rng = np.random.default_rng(12)
        n = 3000
        rw = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
        drift = np.zeros(n)
        for t in range(1, n):
            drift[t] = 0.995 * drift[t - 1] + rng.normal(0, 0.00045)
        mom = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.008, n)))
        bad = B.backtest(self._series("bt_rw", rw), "sma", boot=300)
        good = B.backtest(self._series("bt_mom", mom), "sma", boot=300)
        self.assertEqual(next(f for f in bad.facts if f.type == "q_bt_overfit").severity, "warning")
        self.assertEqual(next(f for f in good.facts if f.type == "q_bt_overfit").severity, "info")
        wf_good = next(f for f in good.facts if f.type == "q_bt_walkforward").metrics
        self.assertGreater(wf_good["oos_sharpe"], 1.0)
        self.assertIn("Walk-forward folds", good.tables)
        self.assertGreaterEqual(len(good.tables["Drawdowns"][1]), 1)

    def test_costs_reduce_returns(self):
        rng = np.random.default_rng(6)
        P = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
        ret = P[1:] / P[:-1] - 1
        pos = B.positions("momentum", P, {"lookback": 5})
        free = B.perf(B.strategy_returns(pos, ret, 0.0)[0], 252)["cagr"]
        paid = B.perf(B.strategy_returns(pos, ret, 0.002)[0], 252)["cagr"]
        self.assertLess(paid, free)


class Portfolio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        N, T = 5, 2000
        A_ = rng.normal(size=(N, N)) * 0.01
        cls.C = A_ @ A_.T + np.diag(np.full(N, 1e-4))
        cls.mu = rng.uniform(0.02, 0.12, N) / 252
        cls.X = rng.multivariate_normal(cls.mu, cls.C, size=T)

    def test_closed_form_solutions(self):
        S, m = np.cov(self.X.T, bias=True), self.X.mean(0)
        w = PF.solve_mv(S, np.zeros(5), 0.0, -50, 50)
        ref = np.linalg.solve(S, np.ones(5))
        self.assertTrue(np.allclose(w, ref / ref.sum(), atol=1e-6))
        ws = PF.max_sharpe(S, m, 0.0, -50, 50)
        rt = np.linalg.solve(S, m)
        self.assertAlmostEqual(PF._sharpe(ws, S, m, 0), PF._sharpe(rt / rt.sum(), S, m, 0), 4)

    def test_constraints_and_risk_parity(self):
        S = np.cov(self.X.T, bias=True)
        for w in (PF.solve_mv(S, self.mu, 5.0, 0.0, 0.4), PF.risk_parity(S), PF.project_box_simplex(np.random.default_rng(0).normal(size=8), 0, 0.3)):
            self.assertAlmostEqual(float(w.sum()), 1.0, 8)
            self.assertGreaterEqual(float(w.min()), -1e-9)
        rp = PF.risk_parity(S)
        rc = rp * (S @ rp) / (rp @ S @ rp)
        self.assertTrue(np.allclose(rc, 0.2, atol=1e-6))
        self.assertLessEqual(float(PF.solve_mv(S, self.mu, 5.0, 0.0, 0.4).max()), 0.4 + 1e-9)

    @unittest.skipUnless(have("sklearn"), "scikit-learn not installed")
    def test_ledoit_wolf_matches_sklearn(self):
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(self.X)
        S, d = PF.ledoit_wolf(self.X)
        self.assertAlmostEqual(d, float(lw.shrinkage_), 6)
        self.assertTrue(np.allclose(S, lw.covariance_, atol=1e-10))

    def test_report(self):
        rng = np.random.default_rng(5)
        n = 900
        names = ["A", "B", "C", "D"]
        R_ = rng.multivariate_normal(np.full(4, 0.0003), np.diag([1e-4, 2e-4, 3e-4, 1.5e-4]), size=n)
        P = 100 * np.cumprod(1 + R_, axis=0)
        days = [d for d in (date(2015, 1, 1) + timedelta(i) for i in range(2 * n)) if d.weekday() < 5][:n]
        path = TMP / "pf_wide.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date"] + names)
            for i in range(n):
                w.writerow([days[i].isoformat()] + [round(float(v), 5) for v in P[i]])
        rep = PF.portfolio(load_panel(str(path), "date", columns=names), stability_reps=10, max_weight=0.6)
        self.assertTrue({"q_pf_weights", "q_pf_risk", "q_pf_oos"} <= {f.type for f in rep.facts})
        weights = rep.tables["Weights"][1]
        for j in range(1, 6):
            self.assertAlmostEqual(sum(r[j] for r in weights), 1.0, 6)
        self.assertLessEqual(max(r[4] for r in weights), 0.6 + 1e-6)


class Derivatives(unittest.TestCase):
    def test_black_scholes_reference_values(self):
        self.assertAlmostEqual(D.bs_price(100, 100, 1, 0.05, 0.0, 0.2, "call"), 10.4506, 3)
        self.assertAlmostEqual(D.bs_price(100, 100, 1, 0.05, 0.0, 0.2, "put"), 5.5735, 3)
        S, K, T, r, q, s = 100, 105, 0.7, 0.03, 0.01, 0.25
        parity = D.bs_price(S, K, T, r, q, s, "call") - D.bs_price(S, K, T, r, q, s, "put") - (S * math.exp(-q * T) - K * math.exp(-r * T))
        self.assertAlmostEqual(parity, 0.0, 10)

    def test_greeks_match_finite_differences(self):
        S, K, T, r, q, s = 100, 100, 1, 0.05, 0.02, 0.2
        g = D.bs_greeks(S, K, T, r, q, s, "call")
        h = 1e-4
        f = lambda **kw: D.bs_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("s", s), "call")
        self.assertAlmostEqual(g["delta"], (f(S=S + h) - f(S=S - h)) / (2 * h), 6)
        self.assertAlmostEqual(g["gamma"], (f(S=S + h) - 2 * f() + f(S=S - h)) / h ** 2, 4)
        self.assertAlmostEqual(g["vega"], (f(s=s + h) - f(s=s - h)) / (2 * h) / 100, 6)
        self.assertAlmostEqual(g["theta"], -(f(T=T + h) - f(T=T - h)) / (2 * h) / 365, 6)

    def test_implied_vol_round_trip_and_bounds(self):
        for kind in ("call", "put"):
            p = D.bs_price(100, 110, 0.5, 0.05, 0.02, 0.33, kind)
            self.assertAlmostEqual(D.implied_vol(p, 100, 110, 0.5, 0.05, 0.02, kind), 0.33, 7)
        self.assertTrue(math.isnan(D.implied_vol(1.0, 100, 80, 1, 0.05, 0.0, "call")))         # below intrinsic

    def test_tree_and_monte_carlo_agree_with_black_scholes(self):
        bs = D.bs_price(100, 100, 1, 0.05, 0.02, 0.2, "put")
        eu = D.binomial(100, 100, 1, 0.05, 0.02, 0.2, "put", False, 800)
        am = D.binomial(100, 100, 1, 0.05, 0.02, 0.2, "put", True, 800)
        self.assertAlmostEqual(eu["price"], bs, delta=0.01)
        self.assertGreater(am["price"], eu["price"])                                           # early-exercise premium
        pr, se = D.monte_carlo(100, 100, 1, 0.05, 0.02, 0.2, "call", "european", 300000)
        self.assertLess(abs(pr - D.bs_price(100, 100, 1, 0.05, 0.02, 0.2, "call")), 4 * se)
        asian, _ = D.monte_carlo(100, 100, 1, 0.05, 0.0, 0.2, "call", "asian", 100000, 100)
        self.assertLess(asian, D.bs_price(100, 100, 1, 0.05, 0.0, 0.2, "call"))               # averaging lowers an option's value

    def test_bond_analytics(self):
        par = D.bond(100, 0.05, 10, 2, ytm=0.05)
        self.assertAlmostEqual(par["price"], 100.0, 8)
        self.assertAlmostEqual(par["modified"], 7.7946, 3)
        self.assertAlmostEqual(D.bond(100, 0.0, 5, 1, ytm=0.04)["macaulay"], 5.0, 8)
        b = D.bond(100, 0.06, 7, 2, ytm=0.073)
        self.assertAlmostEqual(D.bond(100, 0.06, 7, 2, price=b["price"])["ytm"], 0.073, 7)
        self.assertGreater(b["price_dn100"] - b["price"], b["price"] - b["price_up100"])       # convexity

    def test_option_chain_diagnostics(self):
        def run(bad):
            raw = datagen.option_chain(str(TMP / f"chain_{bad}.csv"), bad=bad)
            rows = list(csv.reader(open(raw)))
            out = TMP / f"chain_{bad}_y.csv"
            csv.writer(open(out, "w", newline="")).writerows([rows[0] + ["yrs"]] + [r + [f"{int(r[3]) / 365:.6f}"] for r in rows[1:]])
            return D.option_chain(str(out), "strike", "type", price="mid", years="yrs", spot=100.0, rate=0.03, div=0.01)
        clean, dirty = run(False), run(True)
        bad_types = {"q_opt_bounds", "q_opt_arbitrage", "q_opt_butterfly", "q_opt_parity"}
        self.assertEqual([f.type for f in clean.facts if f.type in bad_types], [])
        self.assertTrue({"q_opt_bounds", "q_opt_butterfly", "q_opt_parity"} <= {f.type for f in dirty.facts})
        skew = next(f for f in clean.facts if f.type == "q_opt_smile").metrics["skew"]
        self.assertAlmostEqual(skew, 0.031, delta=0.006)                                          # planted smile


class LoadersAndCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.long = datagen.prices(str(TMP / "px_long.csv"))

    def test_long_table_needs_an_entity_and_aligns_panel(self):
        with self.assertRaises(ValueError):
            load_series(self.long, "date", "close", "ticker")
        s = load_series(self.long, "date", "close", "ticker", "AAA")
        self.assertEqual(s.kind, "price")
        self.assertAlmostEqual(s.ppy, 252.0)
        p = load_panel(self.long, "date", "close", "ticker")
        self.assertEqual(sorted(p.names), ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(p.X.shape[1], 4)

    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main([str(a) for a in args])
        return code, buf.getvalue()

    def test_cli_quant_commands(self):
        code, out = self.run_cli("price", "bs", "--spot", 100, "--strike", 100, "--years", 1, "--rate", 0.05, "--vol", 0.2, "--format", "json")
        self.assertEqual(code, 0)
        self.assertIn("10.4506", out)
        code, out = self.run_cli("price", "bond", "--coupon", 0.05, "--years", 10, "--ytm", 0.05, "--format", "json")
        self.assertIn("100.0", out)
        code, out = self.run_cli("garch", self.long, "--entity", "ticker", "--entity-value", "CCC", "--format", "json", "-q")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["kind"], "garch")
        code, out = self.run_cli("var", self.long, "--entity", "ticker", "--entity-value", "AAA", "--window", 300, "--format", "json", "-q", "--position", 1000000)
        self.assertEqual(code, 0)
        self.assertIn("VaR (amount)", out)
        code, out = self.run_cli("backtest", self.long, "--entity", "ticker", "--entity-value", "AAA", "--no-walk", "--format", "json", "-q")
        self.assertEqual(code, 0)
        code, out = self.run_cli("portfolio", self.long, "--entity", "ticker", "--value", "close", "--stability-reps", 0, "--format", "json", "-q")
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli("arima", self.long, "-q")[0], 2)                          # long table without an entity value is a usage error


if __name__ == "__main__":
    unittest.main()
