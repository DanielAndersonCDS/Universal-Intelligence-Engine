"""Portfolio construction with evidence: constrained mean-variance (min variance, max Sharpe, efficient frontier), risk parity, equal weight and
inverse volatility; Ledoit-Wolf shrinkage; risk-contribution and concentration diagnostics; weight-stability bootstrap; and an out-of-sample
rebalancing backtest that tests whether optimisation beats 1/N after costs."""
from __future__ import annotations
import math
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from .backtest import block_bootstrap, drawdowns, perf
from .series import Panel

NAMES = {"equal": "Equal weight", "inv_vol": "Inverse volatility", "min_var": "Minimum variance", "max_sharpe": "Maximum Sharpe", "risk_parity": "Risk parity"}


def project_box_simplex(v, lo, hi, total: float = 1.0):
    """Euclidean projection onto {w: sum(w) = total, lo <= w <= hi} by bisection on the shift."""
    v = np.asarray(v, float)
    lo = np.broadcast_to(np.asarray(lo, float), v.shape)
    hi = np.broadcast_to(np.asarray(hi, float), v.shape)
    a, b = float((v - hi).min()) - 1.0, float((v - lo).max()) + 1.0
    for _ in range(80):
        t = (a + b) / 2
        if np.clip(v - t, lo, hi).sum() > total:
            a = t
        else:
            b = t
    return np.clip(v - (a + b) / 2, lo, hi)


def ledoit_wolf(X):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity. Returns (shrunk covariance, intensity)."""
    X = np.asarray(X, float)
    T, N = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    d2 = float(np.sum((S - mu * np.eye(N)) ** 2)) / N
    b2 = min(float(np.sum([np.sum((np.outer(x, x) - S) ** 2) for x in Xc])) / (T * T * N), d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return (1 - delta) * S + delta * mu * np.eye(N), float(delta)


def solve_mv(Sigma, mu, lam, lo, hi, w0=None, iters: int = 4000, tol: float = 1e-11):
    """min w'Σw - lam * mu'w  s.t. sum w = 1, lo <= w <= hi  (accelerated projected gradient)."""
    N = len(mu)
    L = 2 * float(np.linalg.eigvalsh(Sigma)[-1]) + 1e-12
    w = project_box_simplex(np.full(N, 1.0 / N) if w0 is None else w0, lo, hi)
    y, tk = w.copy(), 1.0
    for _ in range(iters):
        g = 2 * Sigma @ y - lam * mu
        wn = project_box_simplex(y - g / L, lo, hi)
        tn = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        y = wn + (tk - 1) / tn * (wn - w)
        if np.max(np.abs(wn - w)) < tol:
            w = wn
            break
        w, tk = wn, tn
    return w


def _sharpe(w, Sigma, mu, rf):
    v = float(w @ Sigma @ w)
    return (float(mu @ w) - rf) / math.sqrt(v) if v > 0 else -1e9


def max_sharpe(Sigma, mu, rf, lo, hi):
    """Maximum-Sharpe point on the constrained frontier: golden-section over the risk-aversion parameter (Sharpe is unimodal along the frontier)."""
    scale = float(np.linalg.eigvalsh(Sigma)[-1]) / max(float(np.max(np.abs(mu - rf))), 1e-9)
    lams = scale * np.geomspace(1e-3, 1e3, 13)
    ws = [solve_mv(Sigma, mu - rf, l, lo, hi) for l in lams]
    sh = [_sharpe(w, Sigma, mu, rf) for w in ws]
    k = int(np.argmax(sh))
    a, b = math.log(lams[max(k - 1, 0)]), math.log(lams[min(k + 1, len(lams) - 1)])
    gr = (math.sqrt(5) - 1) / 2
    f = lambda t: -_sharpe(solve_mv(Sigma, mu - rf, math.exp(t), lo, hi, ws[k]), Sigma, mu, rf)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(18):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    w = solve_mv(Sigma, mu - rf, math.exp((a + b) / 2), lo, hi, ws[k])
    return w if _sharpe(w, Sigma, mu, rf) >= sh[k] - 1e-12 else ws[k]


def risk_parity(Sigma, iters: int = 200):
    """Equal risk contribution weights (long-only) via Spinu's convex formulation, damped Newton."""
    N = Sigma.shape[0]
    y = 1.0 / np.sqrt(np.diag(Sigma))
    y = y / y.sum()
    for _ in range(iters):
        g = Sigma @ y - 1.0 / (N * y)
        H = Sigma + np.diag(1.0 / (N * y * y))
        step = np.linalg.solve(H, g)
        t = 1.0
        f0 = 0.5 * y @ Sigma @ y - np.log(y).sum() / N
        while t > 1e-8:
            yn = y - t * step
            if np.all(yn > 0) and 0.5 * yn @ Sigma @ yn - np.log(yn).sum() / N <= f0 + 1e-4 * t * (-g @ step):
                break
            t *= 0.5
        y = y - t * step
        if np.linalg.norm(g) < 1e-12:
            break
    return y / y.sum()


def weights_for(method, X, rf_p, lo, hi, shrink=True):
    """X: T x N simple returns for the estimation window. Returns weights."""
    N = X.shape[1]
    if method == "equal":
        return project_box_simplex(np.full(N, 1.0 / N), lo, hi)
    S, _ = ledoit_wolf(X) if shrink else (np.cov(X.T, bias=True), 0.0)
    if method == "inv_vol":
        w = 1.0 / np.sqrt(np.diag(S))
        return project_box_simplex(w / w.sum(), lo, hi)
    if method == "min_var":
        return solve_mv(S, np.zeros(N), 0.0, lo, hi)
    if method == "max_sharpe":
        return max_sharpe(S, X.mean(0), rf_p, lo, hi)
    if method == "risk_parity":
        return risk_parity(S)
    raise ValueError(method)


def stats_for(w, Sigma, mu, rf, vols):
    var = float(w @ Sigma @ w)
    sd = math.sqrt(var)
    rc = w * (Sigma @ w) / var if var > 0 else np.zeros_like(w)
    return {"ret": float(mu @ w), "vol": sd, "sharpe": (float(mu @ w) - rf) / sd if sd > 0 else 0.0, "rc": rc, "div_ratio": float(w @ vols) / sd if sd > 0 else float("nan"),
            "eff_n": float(1 / np.sum(w ** 2)), "enb": float(math.exp(-np.sum(rc[rc > 1e-12] * np.log(rc[rc > 1e-12])))) if np.all(rc > -1e-9) else float("nan")}


def oos_backtest(R, methods, lookback, rebalance, cost, lo, hi, rf_p, ppy):
    T, N = R.shape
    out = {m: [] for m in methods}
    turn = {m: [] for m in methods}
    W = {m: np.full(N, 1.0 / N) for m in methods}
    started = False
    for t in range(lookback, T):
        if (t - lookback) % rebalance == 0:
            win = R[t - lookback:t]
            for m in methods:
                try:
                    tgt = weights_for(m, win, rf_p, lo, hi)
                except Exception:
                    tgt = W[m]
                tv = float(np.abs(tgt - W[m]).sum()) if started else 0.0
                turn[m].append(tv)
                W[m] = tgt
            started = True
        for m in methods:
            rp = float(W[m] @ R[t])
            out[m].append(rp - (cost * turn[m][-1] if (t - lookback) % rebalance == 0 else 0.0))
            W[m] = W[m] * (1 + R[t]) / (1 + rp) if 1 + rp != 0 else W[m]
    return {m: np.array(v) for m, v in out.items()}, {m: float(np.sum(v)) for m, v in turn.items()}


def portfolio(panel: Panel, methods=("equal", "inv_vol", "min_var", "max_sharpe", "risk_parity"), rf: float = 0.0, min_weight: float = 0.0, max_weight: float = 1.0,
              lookback: int = 252, rebalance: int = 21, cost_bps: float = 5.0, stability_reps: int = 40, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    dates, R = panel.simple_returns()
    R = np.asarray(R, float)
    T, N = R.shape
    names = panel.names
    ppy = panel.ppy
    if min_weight * N > 1 + 1e-9 or max_weight * N < 1 - 1e-9:
        raise ValueError(f"Weight bounds [{min_weight}, {max_weight}] cannot sum to 1 with {N} assets.")
    lo, hi = min_weight, max_weight
    rf_p = rf / ppy
    S, delta = ledoit_wolf(R)
    Sa, mu_a = S * ppy, R.mean(0) * ppy
    vols = np.sqrt(np.diag(Sa))
    cond = float(np.linalg.cond(np.cov(R.T)))
    progress("Optimising", 15)
    W, ST = {}, {}
    for m in methods:
        W[m] = weights_for(m, R, rf_p, lo, hi)
        ST[m] = stats_for(W[m], Sa, mu_a, rf, vols)
    progress("Efficient frontier", 35)
    scale = float(np.linalg.eigvalsh(Sa)[-1]) / max(float(np.max(np.abs(mu_a - rf))), 1e-9)
    front = []
    for l in scale * np.geomspace(1e-3, 1e2, 25):
        w = solve_mv(Sa, mu_a - rf, l, lo, hi)
        front.append((math.sqrt(float(w @ Sa @ w)), float(mu_a @ w)))
    fr_pts = sorted(set((round(a, 6), round(b, 6)) for a, b in front))
    facts = []
    sql = panel.sql

    def add(type_, title, msg, sev, score, cols=None, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=cols or names))

    ms = "max_sharpe" if "max_sharpe" in methods else methods[0]
    top = sorted(zip(names, W[ms]), key=lambda t: -abs(t[1]))[:4]
    bound = f"long-only, weights between {lo:g} and {hi:g}" if lo >= 0 else f"weights between {lo:g} and {hi:g} (shorting allowed)"
    add("q_pf_weights", f"{NAMES[ms]} portfolio", f"The {NAMES[ms].lower()} portfolio ({bound}; Ledoit-Wolf shrinkage intensity {delta:.2f}) puts " + ", ".join(f"{n} {w:.0%}" for n, w in top) +
        f". In-sample it has expected return {ST[ms]['ret']:.1%}, volatility {ST[ms]['vol']:.1%} and Sharpe {ST[ms]['sharpe']:.2f}. These are estimates from {T:,} periods of history, not forecasts.", "info", 0.5,
        sharpe=ST[ms]["sharpe"], ret=ST[ms]["ret"], vol=ST[ms]["vol"])
    s0 = ST.get("min_var") or ST[ms]
    rc = ST[ms]["rc"]
    ia = int(np.argmax(rc))
    add("q_pf_risk", "Risk concentration", f"In the {NAMES[ms].lower()} portfolio {names[ia]} carries {rc[ia]:.0%} of the risk with {W[ms][ia]:.0%} of the capital. The portfolio behaves like "
        f"{ST[ms]['enb']:.1f} independent bets out of {N} assets (effective number of holdings {ST[ms]['eff_n']:.1f}); diversification ratio {ST[ms]['div_ratio']:.2f}.",
        "notable" if ST[ms]["enb"] < max(2.0, N / 3) else "info", 0.4, enb=ST[ms]["enb"], eff_n=ST[ms]["eff_n"])
    years = T / ppy
    se_mu = float(np.mean(vols) / math.sqrt(years))
    est = []
    if T < 10 * N:
        est.append(f"only {T / N:.0f} observations per asset (rule of thumb: at least 10)")
    if cond > 1e3:
        est.append(f"the sample covariance is ill-conditioned (condition number {cond:.0f})")
    if delta > 0.3:
        est.append(f"shrinkage had to pull the covariance {delta:.0%} toward a diagonal target")
    add("q_pf_estimation", "Estimation error", f"Expected returns are the weakest input: with {years:.1f} years of data each asset's annualized mean return has a standard error of roughly {se_mu:.1%}, "
        f"which is often larger than the spread between assets, so return-chasing weights (maximum Sharpe) are fragile." + (" Also: " + "; ".join(est) + "." if est else ""),
        "notable" if est else "info", 0.5, years=years, se_mean=se_mu)
    stab = None
    if N <= 20 and stability_reps and ms in ("max_sharpe", "min_var"):
        progress("Weight stability bootstrap", 45)
        Ws = []
        rng = np.random.default_rng(3)
        L = max(2, int(1.5 * T ** (1 / 3)))
        for _ in range(stability_reps):
            starts = rng.integers(0, T, size=int(math.ceil(T / L)))
            idx = ((starts[:, None] + np.arange(L)[None, :]) % T).ravel()[:T]
            Ws.append(weights_for(ms, R[idx], rf_p, lo, hi))
        Ws = np.array(Ws)
        stab = (Ws.mean(0), Ws.std(0))
        avg_sd = float(stab[1].mean())
        add("q_pf_stability", "Weight stability", f"Re-estimating the {NAMES[ms].lower()} weights on {stability_reps} block-bootstrap resamples moves each weight by {avg_sd:.0%} on average "
            f"(largest: {names[int(np.argmax(stab[1]))]} +/- {stab[1].max():.0%}). " + ("The weights are not stable enough to trade on their own." if avg_sd > 0.12 else "The weights are reasonably stable."),
            "notable" if avg_sd > 0.12 else "info", min(1.0, avg_sd * 3), avg_sd=avg_sd)
    oos_rows, oos = [], None
    if T >= lookback + 3 * rebalance + 60:
        progress("Out-of-sample rebalancing test", 60)
        series, turns = oos_backtest(R, list(methods), lookback, rebalance, cost_bps / 1e4, lo, hi, rf_p, ppy)
        pf = {m: perf(series[m], ppy, rf) for m in methods}
        eq = series.get("equal", next(iter(series.values())))
        oos = {"series": series, "perf": pf, "turn": turns}
        for m in methods:
            oos_rows.append([NAMES[m], pf[m]["cagr"], pf[m]["vol"], pf[m]["sharpe"], pf[m]["max_dd"], turns[m] * ppy / (len(series[m]) or 1) * rebalance / rebalance, pf[m]["psr"]])
        best = max(methods, key=lambda m: pf[m]["sharpe"])
        verdict = ""
        if "equal" in methods:
            beat = [m for m in methods if m != "equal" and pf[m]["sharpe"] > pf["equal"]["sharpe"]]
            ps = {}
            for m in beat:
                d = series[m] - series["equal"]
                sh, _, _, _ = block_bootstrap(d, 800, ppy=ppy)
                ps[m] = float(np.mean(sh <= 0))
            sig = [m for m in beat if ps[m] < 0.05]
            verdict = (f" Only {', '.join(NAMES[m] for m in sig)} beat{'s' if len(sig) == 1 else ''} equal weight with confidence (bootstrap p < 0.05)." if sig else
                       " No optimized portfolio beats equal weight with statistical confidence; 1/N is a hard benchmark to beat out of sample.")
        add("q_pf_oos", "Out-of-sample comparison", f"Re-optimizing every {rebalance} periods on a {lookback}-period window, with {cost_bps:g} bps costs, the best out-of-sample Sharpe was {NAMES[best]} "
            f"({pf[best]['sharpe']:.2f}); " + "; ".join(f"{NAMES[m]} {pf[m]['sharpe']:.2f}" for m in methods if m != best) + "." + verdict, "info", 0.7, best=best)
    else:
        panel.notes.append(f"Out-of-sample test skipped: needs at least {lookback + 3 * rebalance + 60} periods, got {T}.")
    for i, f in enumerate(facts, 1):
        f.id = f"P{i:03d}"
    charts = {}
    marks = [{"x": float(vols[i]), "y": float(mu_a[i]), "name": names[i]} for i in range(N)] + [{"x": ST[m]["vol"], "y": ST[m]["ret"], "name": NAMES[m]} for m in methods]
    charts[facts[0].id] = ({"kind": "scatter", "xlabel": "volatility (annualized)", "ylabel": "expected return (annualized)", "marks": marks}, fr_pts)
    if oos:
        n_o = len(next(iter(oos["series"].values())))
        od = dates[-n_o:]
        rows = [(od[i],) + tuple(float(np.prod(1 + oos["series"][m][:i + 1])) for m in methods) for i in range(n_o)]
        charts[facts[-1].id if facts[-1].type == "q_pf_oos" else facts[0].id] = ({"kind": "multiline", "names": [NAMES[m] for m in methods], "xlabel": "time", "ylabel": "growth of 1"}, rows)
    w_rows = [[names[i]] + [float(W[m][i]) for m in methods] + [float(vols[i]), float(mu_a[i])] for i in range(N)]
    rc_rows = [[names[i]] + [float(ST[m]["rc"][i]) for m in methods] for i in range(N)]
    summary = {"assets": ", ".join(names[:10]) + (" ..." if N > 10 else ""), "periods": T, "periods per year": ppy, "risk-free (annual)": rf, "weight bounds": f"[{lo:g}, {hi:g}]",
               "covariance": f"Ledoit-Wolf shrinkage, intensity {delta:.2f}", "condition number": cond}
    tables = {"Weights": (["asset"] + [NAMES[m] for m in methods] + ["volatility", "expected return"], w_rows),
              "Portfolio statistics": (["method", "expected return", "volatility", "Sharpe", "diversification ratio", "effective N", "effective bets"],
                                       [[NAMES[m], ST[m]["ret"], ST[m]["vol"], ST[m]["sharpe"], ST[m]["div_ratio"], ST[m]["eff_n"], ST[m]["enb"]] for m in methods]),
              "Risk contributions": (["asset"] + [NAMES[m] for m in methods], rc_rows),
              "Efficient frontier": (["volatility", "expected return"], [list(p) for p in fr_pts])}
    if oos_rows:
        tables["Out-of-sample"] = (["method", "CAGR", "volatility", "Sharpe", "max drawdown", "rebalances", "PSR"], [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in oos_rows])
    if stab:
        tables["Weight stability"] = (["asset", "mean weight", "std of weight"], [[names[i], float(stab[0][i]), float(stab[1][i])] for i in range(N)])
    out = SubReport(kind="portfolio", title="Portfolio optimisation", summary=summary, facts=facts, tables=tables, setup_sql=[panel.source.source_sql()],
                    warnings=list(panel.notes) + ["Optimised weights are estimates from historical data; they are inputs to a decision, not a recommendation."], charts=charts)
    out.provenance = provenance.build([provenance.input_record(panel.source, "data", cfg.hash_data)], {"methods": list(methods), "rf": rf, "min_weight": lo, "max_weight": hi,
                                                                                                        "lookback": lookback, "rebalance": rebalance, "cost_bps": cost_bps}, cfg, None, None, facts, command)
    progress("Done", 100)
    return out
