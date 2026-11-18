"""Backtesting: strategy rules, performance and drawdown analytics, walk-forward optimisation, probabilistic and deflated Sharpe ratios,
block-bootstrap robustness and cost sensitivity.

Timing: a position decided at the close of bar t earns the return of bar t+1 (no look-ahead). Costs are charged on turnover when the position changes.
Every strategy here is causal, so positions computed on the whole series are identical to those computed bar by bar."""
from __future__ import annotations
import itertools
import math
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from .series import Series

GRIDS = {
    "buyhold": {},
    "sma": {"fast": [10, 20, 50], "slow": [100, 150, 200]},
    "momentum": {"lookback": [20, 60, 120, 250]},
    "meanrev": {"lookback": [10, 20, 40], "entry": [1.0, 1.5, 2.0]},
    "rsi": {"period": [7, 14], "low": [25, 30], "high": [60, 70]},
    "breakout": {"lookback": [20, 50, 100]},
}
DEFAULTS = {"buyhold": {}, "sma": {"fast": 20, "slow": 100}, "momentum": {"lookback": 120}, "meanrev": {"lookback": 20, "entry": 1.5},
            "rsi": {"period": 14, "low": 30, "high": 70}, "breakout": {"lookback": 50}}
LABELS = {"buyhold": "Buy and hold", "sma": "Moving-average crossover", "momentum": "Momentum", "meanrev": "Mean reversion (z-score)", "rsi": "RSI mean reversion",
          "breakout": "Donchian breakout", "signal": "User signal column"}


def _sma(x, w):
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _rolling_std(x, w):
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        out[w - 1:] = sliding_window_view(x, w).std(axis=1, ddof=1)
    return out


def positions(strategy: str, prices, params: dict, long_short: bool = False, signal=None) -> np.ndarray:
    """Position held after the close of each bar (decided from data up to and including that bar)."""
    P = np.asarray(prices, float)
    n = len(P)
    short = -1.0 if long_short else 0.0
    pos = np.zeros(n)
    if strategy == "buyhold":
        return np.ones(n)
    if strategy == "signal":
        return np.clip(np.nan_to_num(np.asarray(signal, float)), -1, 1)
    if strategy == "sma":
        f, s = _sma(P, params["fast"]), _sma(P, params["slow"])
        ok = np.isfinite(f) & np.isfinite(s)
        pos[ok] = np.where(f[ok] > s[ok], 1.0, short)
        return pos
    if strategy == "momentum":
        L = params["lookback"]
        m = np.full(n, np.nan)
        m[L:] = P[L:] / P[:-L] - 1
        ok = np.isfinite(m)
        pos[ok] = np.where(m[ok] > 0, 1.0, short)
        return pos
    if strategy == "meanrev":
        L, k = params["lookback"], params["entry"]
        ma, sd = _sma(P, L), _rolling_std(P, L)
        z = (P - ma) / sd
        cur = 0.0
        for t in range(n):
            if not np.isfinite(z[t]):
                continue
            if cur == 0:
                cur = 1.0 if z[t] < -k else (short if (long_short and z[t] > k) else 0.0)
            elif cur > 0 and z[t] >= 0:
                cur = 0.0
            elif cur < 0 and z[t] <= 0:
                cur = 0.0
            pos[t] = cur
        return pos
    if strategy == "rsi":
        L, lo, hi = int(params["period"]), params["low"], params["high"]
        d = np.diff(P, prepend=P[0])
        up, dn = np.maximum(d, 0), np.maximum(-d, 0)
        au = ad = 0.0
        rsi = np.full(n, np.nan)
        for t in range(1, n):
            if t <= L:
                au += up[t] / L
                ad += dn[t] / L
            else:
                au, ad = (au * (L - 1) + up[t]) / L, (ad * (L - 1) + dn[t]) / L
            if t >= L:
                rsi[t] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
        cur = 0.0
        for t in range(n):
            if not np.isfinite(rsi[t]):
                continue
            if cur == 0:
                cur = 1.0 if rsi[t] < lo else (short if (long_short and rsi[t] > hi) else 0.0)
            elif cur > 0 and rsi[t] > hi:
                cur = 0.0
            elif cur < 0 and rsi[t] < lo:
                cur = 0.0
            pos[t] = cur
        return pos
    if strategy == "breakout":
        L = int(params["lookback"])
        cur = 0.0
        for t in range(L, n):
            hi_, lo_ = P[t - L:t].max(), P[t - max(L // 2, 2):t].min()
            if cur == 0 and P[t] > hi_:
                cur = 1.0
            elif cur > 0 and P[t] < lo_:
                cur = 0.0
            pos[t] = cur
        return pos
    raise ValueError(f"unknown strategy '{strategy}'")


def strategy_returns(pos, ret, cost: float = 0.0, vol_target: float | None = None, ppy: float = 252.0, max_lev: float = 3.0, vol_window: int = 60):
    """Strategy return per bar for bars 1..n-1. `ret` are the simple asset returns for those bars; pos[t-1] is applied to ret[t-1]."""
    p = np.asarray(pos[:-1], float).copy()
    ret = np.asarray(ret, float)
    if vol_target:
        lev = np.ones(len(p))
        for k in range(len(p)):
            w = ret[max(0, k - vol_window):k]
            if len(w) >= 20 and w.std() > 0:
                lev[k] = min(max_lev, vol_target / (w.std(ddof=1) * math.sqrt(ppy)))
        p = p * lev
    turnover = np.abs(np.diff(p, prepend=0.0))
    return p * ret - cost * turnover, p, turnover


def drawdowns(r, top: int = 5):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    eps, i, n = [], 0, len(r)
    while i < n:
        if dd[i] < 0:
            start = i - 1 if i > 0 else 0                    # index of the peak bar
            j = i
            while j < n and dd[j] < 0:
                j += 1
            seg = dd[i:j]
            tr = i + int(np.argmin(seg))
            eps.append({"peak": start, "trough": tr, "recovery": j if j < n else None, "depth": float(seg.min()), "to_trough": tr - start,
                        "to_recover": (j - tr) if j < n else None, "length": j - start})
            i = j
        else:
            i += 1
    eps.sort(key=lambda e: e["depth"])
    return {"dd": dd, "equity": eq, "episodes": eps[:top], "max_dd": float(dd.min()) if n else 0.0, "ulcer": float(np.sqrt(np.mean(dd ** 2))),
            "underwater": float(np.mean(dd < 0)), "longest_underwater": max([e["length"] for e in eps], default=0), "avg_dd": float(dd[dd < 0].mean()) if np.any(dd < 0) else 0.0}


def perf(r, ppy: float, rf: float = 0.0, pos=None, turnover=None):
    r = np.asarray(r, float)
    n = len(r)
    if n < 2 or r.std() == 0:
        return {"n": n, "total": float(np.prod(1 + r) - 1) if n else 0.0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "calmar": 0.0,
                "skew": 0.0, "kurt": 3.0, "hit": 0.0, "profit_factor": float("nan"), "avg_win": 0.0, "avg_loss": 0.0, "best": float(r.max()) if n else 0.0, "worst": float(r.min()) if n else 0.0,
                "exposure": float(np.mean(np.abs(pos))) if pos is not None else float("nan"), "turnover_py": float("nan"), "psr": float("nan")}
    eq = np.prod(1 + r)
    years = n / ppy
    cagr = float(eq ** (1 / years) - 1) if eq > 0 else -1.0
    rfp = rf / ppy
    ex = r - rfp
    sh = float(ex.mean() / r.std(ddof=1) * math.sqrt(ppy))
    dsd = math.sqrt(float(np.mean(np.minimum(ex, 0) ** 2)))
    so = float(ex.mean() / dsd * math.sqrt(ppy)) if dsd > 0 else float("nan")
    dd = drawdowns(r)
    z = (r - r.mean()) / r.std()
    sk, ku = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    sr = float(ex.mean() / r.std(ddof=1))
    denom = 1 - sk * sr + (ku - 1) / 4 * sr * sr
    psr = float(stats.norm_cdf(sr * math.sqrt(n - 1) / math.sqrt(denom))) if denom > 0 else float("nan")
    nz = r[r != 0]
    gains, losses = nz[nz > 0], nz[nz < 0]
    return {"n": n, "total": float(eq - 1), "cagr": cagr, "vol": float(r.std(ddof=1) * math.sqrt(ppy)), "sharpe": sh, "sortino": so, "max_dd": dd["max_dd"],
            "calmar": float(cagr / abs(dd["max_dd"])) if dd["max_dd"] < 0 else float("nan"), "skew": sk, "kurt": ku, "hit": float(np.mean(nz > 0)) if len(nz) else 0.0,
            "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"), "avg_win": float(gains.mean()) if len(gains) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0, "best": float(r.max()), "worst": float(r.min()),
            "exposure": float(np.mean(np.abs(pos))) if pos is not None else float("nan"), "turnover_py": float(np.mean(turnover) * ppy) if turnover is not None else float("nan"), "psr": psr}


def deflated_sharpe(sr_best: float, n: int, skew: float, kurt: float, trial_srs, n_trials: int | None = None):
    """Bailey and Lopez de Prado (2014): probability that the best of N trials has a true Sharpe above zero, accounting for selection.
    All Sharpe ratios are per-period (not annualised); `kurt` is non-excess."""
    trial_srs = np.asarray(trial_srs, float)
    N = n_trials or len(trial_srs)
    if N < 2 or trial_srs.std() == 0:
        return float("nan"), 0.0
    g = 0.5772156649
    sr0 = float(trial_srs.std(ddof=1) * ((1 - g) * stats.norm_ppf(1 - 1 / N) + g * stats.norm_ppf(1 - 1 / (N * math.e))))
    denom = 1 - skew * sr_best + (kurt - 1) / 4 * sr_best ** 2
    if denom <= 0:
        return float("nan"), sr0
    return float(stats.norm_cdf((sr_best - sr0) * math.sqrt(n - 1) / math.sqrt(denom))), sr0


def block_bootstrap(r, reps: int = 2000, seed: int = 7, ppy: float = 252.0):
    """Circular block bootstrap of the return series. Returns arrays of Sharpe, max drawdown and terminal wealth."""
    r = np.asarray(r, float)
    n = len(r)
    rng = np.random.default_rng(seed)
    L = max(2, int(math.ceil(1.5 * n ** (1 / 3))))
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n, size=(reps, nb))
    idx = ((starts[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(reps, -1)[:, :n]
    S = r[idx]
    sd = S.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, S.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(ppy), 0.0)
    eq = np.cumprod(1 + S, axis=1)
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    return sharpe, mdd, eq[:, -1] - 1, L


def _grid_combos(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])] or [{}]


def _valid(strategy, c):
    return not (strategy == "sma" and c["fast"] >= c["slow"])


def walk_forward(prices, ret, strategy, grid, train, test, anchored, cost, long_short, ppy, rf, vol_target, objective="sharpe", progress=lambda s, p=0: None):
    combos = [c for c in _grid_combos(grid) if _valid(strategy, c)]
    rets = []
    for c in combos:
        pos = positions(strategy, prices, c, long_short)
        rets.append(strategy_returns(pos, ret, cost, vol_target, ppy)[0])
    R = np.array(rets)
    n = R.shape[1]

    def obj(x):
        if x.std() == 0:
            return -1e9
        if objective == "calmar":
            p_ = perf(x, ppy, rf)
            return p_["calmar"] if np.isfinite(p_["calmar"]) else -1e9
        return float((x.mean() - rf / ppy) / x.std(ddof=1) * math.sqrt(ppy))
    folds, s = [], 0
    while s + train + test <= n:
        a, b, c_ = (0 if anchored else s), s + train, s + train + test
        scores = [obj(R[k, a:b]) for k in range(len(combos))]
        k = int(np.argmax(scores))
        folds.append({"fold": len(folds) + 1, "train": (a, b), "test": (b, c_), "combo": combos[k], "k": k, "is_sharpe": scores[k], "oos_sharpe": obj(R[k, b:c_]),
                      "oos_ret": R[k, b:c_], "median_is": float(np.median(scores))})
        s += test
    return combos, R, folds


def backtest(series: Series, strategy: str = "sma", params: dict | None = None, grid: dict | None = None, long_short: bool = False, cost_bps: float = 5.0, slippage_bps: float = 0.0,
             vol_target: float | None = None, rf: float = 0.0, train: int = 750, test: int = 125, anchored: bool = False, walk: bool = True, signal=None, objective: str = "sharpe",
             boot: int = 2000, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    if series.kind != "price" and strategy != "signal":
        raise ValueError("Strategy backtests need a price series (not returns).")
    P = np.asarray(series.values, float)
    dates = series.dates
    ret = P[1:] / P[:-1] - 1
    ppy = series.ppy
    cost = (cost_bps + slippage_bps) / 1e4
    params = {**DEFAULTS.get(strategy, {}), **(params or {})}
    pos = positions(strategy, P, params, long_short, signal)
    sr_, pos_used, turn = strategy_returns(pos, ret, cost, vol_target, ppy)
    bh = ret.copy()
    p_s, p_b = perf(sr_, ppy, rf, pos_used, turn), perf(bh, ppy, rf, np.ones(len(bh)), None)
    dd = drawdowns(sr_)
    facts = []
    sql = series.sql

    def add(type_, title, msg, sev, score, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=[series.name]))

    lab = LABELS.get(strategy, strategy) + (" " + ", ".join(f"{k}={v}" for k, v in params.items()) if params else "")
    add("q_bt_perf", "Performance", f"{lab}: CAGR {p_s['cagr']:.1%}, volatility {p_s['vol']:.1%}, Sharpe {p_s['sharpe']:.2f}, maximum drawdown {p_s['max_dd']:.1%} over {len(sr_):,} periods "
        f"(buy and hold: CAGR {p_b['cagr']:.1%}, Sharpe {p_b['sharpe']:.2f}, maximum drawdown {p_b['max_dd']:.1%}). Costs of {cost_bps + slippage_bps:g} bps per unit turnover are included; "
        f"exposure {p_s['exposure']:.0%}, turnover {p_s['turnover_py']:.1f}x per year. Probabilistic Sharpe ratio (chance the true Sharpe exceeds 0): {p_s['psr']:.0%}.",
        "info", 0.5, **{k: v for k, v in p_s.items() if isinstance(v, float)})
    if dd["episodes"]:
        e = dd["episodes"][0]
        rec = f"recovered after {e['to_recover']} more periods" if e["to_recover"] is not None else "not yet recovered"
        add("q_bt_drawdown", "Drawdown analysis", f"The worst drawdown is {e['depth']:.1%}: peak {dates[e['peak'] + 1] if e['peak'] + 1 < len(dates) else dates[-1]}, trough {dates[e['trough'] + 1]}, "
            f"{e['to_trough']} periods down, {rec}. The strategy spent {dd['underwater']:.0%} of the time below a previous high (longest {dd['longest_underwater']} periods; "
            f"Ulcer index {dd['ulcer']:.1%}).", "notable" if e["depth"] < -0.3 else "info", min(1.0, abs(e["depth"])), depth=e["depth"], underwater=dd["underwater"])
    cost_rows = []
    for bps in (0, 2, 5, 10, 20, 50):
        s2 = strategy_returns(pos, ret, bps / 1e4, vol_target, ppy)[0]
        cost_rows.append([bps, perf(s2, ppy, rf)["sharpe"], perf(s2, ppy, rf)["cagr"]])
    be = next((c[0] for c in cost_rows if c[1] <= 0), None)
    if p_s["turnover_py"] == p_s["turnover_py"] and p_s["turnover_py"] > 0 and cost_rows[0][1] > 0:
        add("q_bt_costs", "Cost sensitivity", f"Sharpe falls from {cost_rows[0][1]:.2f} with no costs to {cost_rows[-2][1]:.2f} at 20 bps per unit turnover"
            f"{'' if be is None else f' and is no longer positive at {be} bps'}; turnover is {p_s['turnover_py']:.1f}x per year.", "notable" if be is not None and be <= 20 else "info", 0.4)
    charts = {}
    ed = dates[1:]
    eq_s, eq_b = dd["equity"], np.cumprod(1 + bh)
    charts["__perf"] = ({"kind": "multiline", "names": ["strategy", "buy and hold"], "xlabel": "time", "ylabel": "growth of 1", "logy": True},
                        [(ed[i], float(eq_s[i]), float(eq_b[i])) for i in range(len(ed))][-3000:])
    tables = {}
    wf = None
    grid = grid if grid is not None else GRIDS.get(strategy, {})
    if walk and grid and strategy not in ("buyhold", "signal"):
        combos_probe = [c for c in _grid_combos(grid) if _valid(strategy, c)]
        need = max([v for c in combos_probe for k, v in c.items() if k in ("slow", "lookback", "period")] + [20])
        if train < need + 60:
            train = need + 100
        if len(ret) >= train + test:
            progress("Walk-forward", 30)
            combos, R, folds = walk_forward(P, ret, strategy, grid, train, test, anchored, cost, long_short, ppy, rf, vol_target, objective, progress)
            if folds:
                oos = np.concatenate([f["oos_ret"] for f in folds])
                start = folds[0]["test"][0]
                p_o = perf(oos, ppy, rf)
                bh_o = perf(bh[start:start + len(oos)], ppy, rf)
                is_mean = float(np.mean([f["is_sharpe"] for f in folds]))
                wfe = p_o["sharpe"] / is_mean if is_mean > 0 else float("nan")
                share_pos = float(np.mean([f["oos_sharpe"] > 0 for f in folds]))
                keys = list(grid)
                stab = float(np.mean([max(np.mean([f["combo"][k] == v for f in folds]) for v in set(f["combo"][k] for f in folds)) for k in keys])) if keys else 1.0
                full_sh = np.array([(x.mean() - rf / ppy) / x.std(ddof=1) if x.std() > 0 else 0.0 for x in R])
                kbest = int(np.argmax(full_sh))
                pb = perf(R[kbest], ppy, rf)
                dsr, sr0 = deflated_sharpe(full_sh[kbest], len(R[kbest]), pb["skew"], pb["kurt"], full_sh)
                wf = {"folds": folds, "oos": oos, "p_o": p_o, "bh_o": bh_o, "wfe": wfe, "is": is_mean, "stab": stab, "dsr": dsr, "best": combos[kbest], "best_sh": pb["sharpe"], "n": len(combos)}
                add("q_bt_walkforward", "Walk-forward result", f"Across {len(folds)} {'anchored' if anchored else 'rolling'} folds (train {train}, test {test} periods) the parameters chosen in-sample "
                    f"had an average in-sample Sharpe of {is_mean:.2f}; stitched out-of-sample Sharpe was {p_o['sharpe']:.2f} (CAGR {p_o['cagr']:.1%}, maximum drawdown {p_o['max_dd']:.1%}) "
                    f"against {bh_o['sharpe']:.2f} for buy and hold over the same period. Walk-forward efficiency (OOS / IS Sharpe) is "
                    f"{'n/a' if wfe != wfe else format(wfe, '.2f')}; {share_pos:.0%} of folds were profitable out of sample.", "info" if p_o["sharpe"] > 0 else "notable", 0.6,
                    oos_sharpe=p_o["sharpe"], is_sharpe=is_mean, wfe=wfe, folds=len(folds))
                issues = []
                if p_o["sharpe"] <= 0:
                    issues.append(f"the out-of-sample Sharpe is {p_o['sharpe']:.2f}")
                if wfe == wfe and wfe < 0.5:
                    issues.append(f"walk-forward efficiency is only {wfe:.2f} (out-of-sample performance is less than half of in-sample)")
                if stab < 0.5:
                    issues.append(f"the chosen parameters change from fold to fold (stability {stab:.0%}), a sign the optimum is noise")
                if dsr == dsr and dsr < 0.95:
                    issues.append(f"the best of {len(combos)} full-sample parameter sets has a deflated Sharpe probability of only {dsr:.0%}, so its in-sample edge is not distinguishable from selection luck")
                if p_o["sharpe"] <= bh_o["sharpe"]:
                    issues.append("it does not beat buy and hold out of sample on risk-adjusted terms")
                if issues:
                    add("q_bt_overfit", "Overfitting evidence", "Warning: " + "; ".join(issues) + ".", "warning" if len(issues) >= 2 or p_o["sharpe"] <= 0 else "notable", 0.8, issues=len(issues))
                else:
                    add("q_bt_overfit", "Overfitting checks", f"No overfitting flags: out-of-sample Sharpe {p_o['sharpe']:.2f} beats buy and hold, efficiency {wfe:.2f}, parameters stable ({stab:.0%}), "
                        f"deflated Sharpe probability {dsr:.0%} for the best of {len(combos)} sets. This does not prove the edge will persist.", "info", 0.3)
                sh_b, mdd_b, tw_b, L = block_bootstrap(oos, boot, ppy=ppy)
                add("q_bt_robust", "Bootstrap robustness", f"A circular block bootstrap ({boot:,} resamples, block length {L}) of the out-of-sample returns gives a 5th-95th percentile Sharpe of "
                    f"{np.percentile(sh_b, 5):.2f} to {np.percentile(sh_b, 95):.2f}; the Sharpe is at or below zero in {np.mean(sh_b <= 0):.0%} of resamples, and the 95th-percentile "
                    f"maximum drawdown is {np.percentile(mdd_b, 5):.1%}.", "notable" if np.mean(sh_b <= 0) > 0.2 else "info", float(np.mean(sh_b <= 0)),
                    p_sharpe_nonpos=float(np.mean(sh_b <= 0)))
                fold_rows = [[f["fold"], str(dates[f["train"][0] + 1]) + " to " + str(dates[f["train"][1]]), str(dates[f["test"][0] + 1]) + " to " + str(dates[min(f["test"][1], len(dates) - 1)]),
                              ", ".join(f"{k}={v}" for k, v in f["combo"].items()), f["is_sharpe"], f["oos_sharpe"]] for f in folds]
                tables["Walk-forward folds"] = (["fold", "train", "test", "chosen parameters", "IS Sharpe", "OOS Sharpe"], fold_rows)
                eo = np.cumprod(1 + oos)
                ebh = np.cumprod(1 + bh[start:start + len(oos)])
                od = dates[1:][start:start + len(oos)]
                charts["__wf"] = ({"kind": "multiline", "names": ["walk-forward OOS", "buy and hold"], "xlabel": "time", "ylabel": "growth of 1",
                                   "markers": [dates[1:][f["test"][0]] for f in folds[1:]]}, [(od[i], float(eo[i]), float(ebh[i])) for i in range(len(od))])
        else:
            series.notes.append(f"Walk-forward skipped: needs at least {train + test} returns, got {len(ret)}.")
    perf_rows = [[n_, p["cagr"], p["vol"], p["sharpe"], p["sortino"], p["max_dd"], p["calmar"], p["hit"], p["profit_factor"], p["psr"]] for n_, p in
                 (("Strategy (fixed params)", p_s), ("Buy and hold", p_b)) + ((("Walk-forward OOS", wf["p_o"]), ("Buy and hold (same period)", wf["bh_o"])) if wf else ())]
    tables = {"Performance": (["", "CAGR", "volatility", "Sharpe", "Sortino", "max drawdown", "Calmar", "hit rate", "profit factor", "PSR"], perf_rows),
              "Drawdowns": (["rank", "peak", "trough", "recovery", "depth", "periods to trough", "periods to recover"],
                            [[i + 1, str(dates[e["peak"] + 1] if e["peak"] + 1 < len(dates) else dates[-1]), str(dates[e["trough"] + 1]),
                              str(dates[e["recovery"] + 1]) if e["recovery"] is not None and e["recovery"] + 1 < len(dates) else "not recovered", e["depth"], e["to_trough"], e["to_recover"]]
                             for i, e in enumerate(dd["episodes"])]),
              "Cost sensitivity": (["cost (bps per unit turnover)", "Sharpe", "CAGR"], cost_rows), **tables}
    for i, f in enumerate(facts, 1):
        f.id = f"B{i:03d}"
    mapping = {"__perf": facts[0].id, "__wf": next((f.id for f in facts if f.type == "q_bt_walkforward"), facts[0].id)}
    charts = {mapping[k] if k in mapping else k: v for k, v in charts.items()}
    if "__dd" not in charts and dd["episodes"]:
        did = next(f.id for f in facts if f.type == "q_bt_drawdown")
        charts[did] = ({"kind": "multiline", "names": ["strategy drawdown", "buy and hold drawdown"], "xlabel": "time", "ylabel": "drawdown"},
                       [(ed[i], float(dd["dd"][i]), float(drawdowns(bh)["dd"][i])) for i in range(len(ed))][-3000:])
    summary = {"series": series.name, "strategy": lab, "periods": len(sr_), "periods per year": ppy, "costs (bps)": cost_bps + slippage_bps, "long/short": long_short,
               "volatility target": vol_target or "none", "risk-free (annual)": rf, "walk-forward": ("yes" if wf else "no")}
    out = SubReport(kind="backtest", title=f"Backtest: {LABELS.get(strategy, strategy)} on {series.name}", summary=summary, facts=facts, tables=tables, setup_sql=[series.source.source_sql()],
                    warnings=list(series.notes) + ["Backtests are historical simulations: they ignore survivorship bias, taxes, capacity and regime change, and do not predict future results."], charts=charts)
    out.provenance = provenance.build([provenance.input_record(series.source, "data", cfg.hash_data)], {"strategy": strategy, "params": params, "grid": grid, "cost_bps": cost_bps,
                                                                                                         "slippage_bps": slippage_bps, "long_short": long_short, "vol_target": vol_target,
                                                                                                         "train": train, "test": test, "anchored": anchored, "objective": objective},
                                      cfg, None, None, facts, command)
    progress("Done", 100)
    return out
