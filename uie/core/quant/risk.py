"""Value at Risk and Expected Shortfall: historical, normal, Student-t, Cornish-Fisher, EWMA (and optionally GARCH-filtered historical
simulation), with rolling out-of-sample backtests (Kupiec, Christoffersen, Basel traffic light, ES ratio) and worst-period stress table."""
from __future__ import annotations
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from . import optim
from .garch import ewma_variance, fit_garch, next_variance
from .series import Series

METHODS = {"historical": "Historical", "normal": "Normal", "t": "Student-t", "cornish": "Cornish-Fisher", "ewma": "EWMA (0.94)", "garch": "GARCH-FHS"}


def t_pdf(x, nu):
    return math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi) * (1 + x * x / nu) ** (-(nu + 1) / 2)


def fit_t(r):
    """MLE of a location-scale Student-t (mu, scale, nu)."""
    r = np.asarray(r, float)
    sd = r.std()

    def nll(x):
        mu, s, nu = x[0], math.exp(min(max(x[1], -50), 50)), 2.05 + math.exp(min(max(x[2], -50), 50))
        z = (r - mu) / s
        ll = len(r) * (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(s)) - (nu + 1) / 2 * np.sum(np.log1p(z * z / nu))
        return -ll if np.isfinite(ll) else 1e12
    best = None
    for nu0 in (4.0, 10.0):
        res = optim.minimize(nll, [r.mean(), math.log(sd * math.sqrt((nu0 - 2) / nu0)), math.log(max(nu0 - 2.05, 0.1))], maxiter=100)
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), math.exp(min(max(best.x[1], -50), 50)), 2.05 + math.exp(min(max(best.x[2], -50), 50))


def var_es(method: str, r, alpha: float):
    """One-period VaR and ES as positive loss fractions, from a return sample."""
    r = np.asarray(r, float)
    if method == "historical":
        q = float(np.quantile(-r, alpha))
        tail = -r[-r >= q] if False else (-r)[(-r) >= q]
        return q, float(tail.mean()) if len(tail) else q
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if method == "normal":
        z = stats.norm_ppf(alpha)
        return -mu + z * sd, -mu + sd * stats.norm_pdf(z) / (1 - alpha)
    if method == "t":
        m, s, nu = fit_t(r)
        q = stats.t_ppf(alpha, nu)
        return -m + s * q, -m + s * t_pdf(q, nu) / (1 - alpha) * (nu + q * q) / (nu - 1)
    if method == "cornish":
        z = stats.norm_ppf(alpha)
        sk = float(np.mean(((r - mu) / sd) ** 3))
        ku = float(np.mean(((r - mu) / sd) ** 4)) - 3
        zc = z + (z * z - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk * sk / 36
        v = -mu + zc * sd
        return v, v + sd * (stats.norm_pdf(z) / (1 - alpha) - z)
    if method == "ewma":
        sg = math.sqrt(0.94 * float(ewma_variance(r)[-1]) + 0.06 * float(r[-1] - r.mean()) ** 2)
        z = stats.norm_ppf(alpha)
        return z * sg, sg * stats.norm_pdf(z) / (1 - alpha)
    raise ValueError(method)


def rolling_var(method: str, r, alpha: float, window: int, refit: int = 50):
    """Forecast VaR/ES for each t >= window using only data before t. Returns (var, es) arrays aligned to r[window:]."""
    n = len(r)
    if method == "historical":
        W = sliding_window_view(-r, window)[:n - window]
        q = np.quantile(W, alpha, axis=1)
        es = np.array([w[w >= qq].mean() for w, qq in zip(W, q)])
        return q, es
    if method in ("normal", "t", "cornish"):
        W = sliding_window_view(r, window)[:n - window]
        mu, sd = W.mean(1), W.std(1, ddof=1)
        z = stats.norm_ppf(alpha)
        if method == "normal":
            return -mu + z * sd, -mu + sd * stats.norm_pdf(z) / (1 - alpha)
        zs = (W - mu[:, None]) / sd[:, None]
        sk, ku = (zs ** 3).mean(1), (zs ** 4).mean(1) - 3
        if method == "cornish":
            zc = z + (z * z - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk * sk / 36
            v = -mu + zc * sd
            return v, v + sd * (stats.norm_pdf(z) / (1 - alpha) - z)
        nu = np.clip(4 + 6 / np.maximum(ku, 0.05), 4.5, 60)           # moment estimate of the degrees of freedom (fast; the headline VaR uses MLE)
        q = np.array([stats.t_ppf(alpha, v) for v in nu])
        s = sd * np.sqrt((nu - 2) / nu)
        pdf = np.array([t_pdf(qq, v) for qq, v in zip(q, nu)])
        return -mu + s * q, -mu + s * pdf / (1 - alpha) * (nu + q * q) / (nu - 1)
    if method == "ewma":
        sg = np.sqrt(ewma_variance(r))
        z = stats.norm_ppf(alpha)
        sg = sg[window:]
        return z * sg, sg * stats.norm_pdf(z) / (1 - alpha)
    if method == "garch":
        var, es = np.empty(n - window), np.empty(n - window)
        z0 = None
        for s in range(0, n - window, refit):
            seg = r[s:s + window] * 100
            f = fit_garch(seg, "gjr", "t")
            zs = f["z"]
            qz, tail = float(np.quantile(-zs, alpha)), None
            tail = float(np.mean((-zs)[(-zs) >= qz]))
            rr = r * 100
            for t in range(s + window, min(s + window + refit, n)):
                # one-step variance using parameters fixed at the last refit and data up to t-1
                seg2 = rr[t - window:t]
                e = seg2 - f["mu"]
                s2 = np.empty(window)
                from .garch import variance_path
                s2 = variance_path(seg2, f["mu"], f["omega"], f["alpha"], f["beta"], f["gamma"])
                nxt = f["omega"] + (f["alpha"] + f["gamma"] * (e[-1] < 0)) * e[-1] ** 2 + f["beta"] * s2[-1]
                sg = math.sqrt(nxt) / 100
                var[t - window], es[t - window] = -f["mu"] / 100 + sg * qz, -f["mu"] / 100 + sg * tail
        return var, es
    raise ValueError(method)


def kupiec(x: int, N: int, p: float):
    """Unconditional coverage LR test; p = expected exception probability (1 - confidence)."""
    if N == 0:
        return float("nan"), float("nan")
    ph = x / N
    ll0 = (N - x) * math.log(1 - p) + (x * math.log(p) if x else 0.0)
    ll1 = ((N - x) * math.log(1 - ph) if x < N else 0.0) + (x * math.log(ph) if x else 0.0)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, stats.chi2_sf(lr, 1)


def christoffersen(exc):
    exc = np.asarray(exc, int)
    a, b = exc[:-1], exc[1:]
    n00, n01, n10, n11 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    tot = n00 + n01 + n10 + n11
    if n01 + n11 == 0 or tot == 0:
        return float("nan"), float("nan")
    pi = (n01 + n11) / tot
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    def term(c, p_):
        return c * math.log(p_) if c and p_ > 0 else 0.0
    ll0 = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    ll1 = term(n00, 1 - pi01) + term(n01, pi01) + term(n10, 1 - pi11) + term(n11, pi11)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, stats.chi2_sf(lr, 1)


def traffic_light(x250: int) -> str:
    return "green" if x250 <= 4 else "yellow" if x250 <= 9 else "red"


def backtest_method(method, r, alpha, window, refit=50):
    var, es = rolling_var(method, r, alpha, window, refit)
    act = -r[window:]
    exc = (act > var).astype(int)
    N, x = len(exc), int(exc.sum())
    lr_pof, p_pof = kupiec(x, N, 1 - alpha)
    lr_ind, p_ind = christoffersen(exc)
    lr_cc = lr_pof + (lr_ind if np.isfinite(lr_ind) else 0.0)
    p_cc = stats.chi2_sf(lr_cc, 2)
    es_ratio = float(np.mean(act[exc == 1]) / np.mean(es[exc == 1])) if x else float("nan")
    last = int(exc[-250:].sum())
    return {"var": var, "es": es, "exc": exc, "N": N, "x": x, "expected": N * (1 - alpha), "rate": x / N, "p_pof": p_pof, "p_ind": p_ind, "p_cc": p_cc, "es_ratio": es_ratio,
            "last250": last, "light": traffic_light(last) if alpha >= 0.985 else None}


def risk(series: Series, confidence=(0.95, 0.99), methods=("historical", "normal", "t", "cornish", "ewma"), window: int = 500, horizon: int = 1, position: float | None = None,
         refit: int = 50, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    dates, r = series.simple_returns()
    r = np.asarray(r, float)
    ok = np.isfinite(r)
    dates, r = [d for d, k in zip(dates, ok) if k], r[ok]
    if len(r) < window + 100:
        window = max(120, len(r) // 2)
    if len(r) < 250:
        raise ValueError(f"VaR needs at least 250 returns; got {len(r)}.")
    methods = [m for m in methods if m in METHODS]
    money = (lambda v: f" (about {v * position:,.0f} on a position of {position:,.0f})") if position else (lambda v: "")
    sq = math.sqrt(horizon)
    sql = series.sql
    facts, rows_now, rows_bt = [], [], []
    cur = {}
    for m in methods:
        if m == "garch":
            continue
        for a in confidence:
            v, e = var_es(m, r[-window:], a)
            cur[(m, a)] = (v * sq, e * sq)
            rows_now.append([METHODS[m], a, v * sq, e * sq] + ([v * sq * position, e * sq * position] if position else []))
    if "garch" in methods:
        f = fit_garch(r * 100, "gjr", "t")
        qz = {a: float(np.quantile(-f["z"], a)) for a in confidence}
        sg = math.sqrt(next_variance(f, r * 100)) / 100
        for a in confidence:
            tail = float(np.mean((-f["z"])[(-f["z"]) >= qz[a]]))
            v, e = -f["mu"] / 100 + sg * qz[a], -f["mu"] / 100 + sg * tail
            cur[("garch", a)] = (v * sq, e * sq)
            rows_now.append([METHODS["garch"], a, v * sq, e * sq] + ([v * sq * position, e * sq * position] if position else []))
    bts = {}
    for k, m in enumerate(methods):
        for a in confidence:
            progress(f"Backtesting {METHODS[m]} at {a:.0%}", 15 + int(70 * (k * len(confidence) + confidence.index(a)) / (len(methods) * len(confidence))))
            try:
                bts[(m, a)] = backtest_method(m, r, a, window, refit)
            except Exception as ex:                       # one method failing must not stop the report
                series.notes.append(f"{METHODS[m]} backtest failed at {a:.0%}: {ex}")
    for (m, a), b in bts.items():
        rows_bt.append([METHODS[m], a, b["N"], b["x"], b["expected"], b["rate"], b["p_pof"], b["p_ind"], b["p_cc"], b["es_ratio"], b["light"] or ""])
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sk = float(np.mean(((r - mu) / sd) ** 3))
    ku = float(np.mean(((r - mu) / sd) ** 4)) - 3

    def add(type_, title, msg, sev, score, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=[series.name]))

    hl = max(confidence)
    base = "historical" if "historical" in methods else methods[0]
    v, e = cur.get((base, hl), (float("nan"), float("nan")))
    hz = f"{horizon}-period" if horizon > 1 else "1-period"
    txt = (f"The {hl:.0%} {hz} {METHODS[base].lower()} VaR is {v:.2%} of the position{money(v)} and the expected shortfall is {e:.2%}{money(e)} "
           f"(estimated on the last {min(window, len(r)):,} returns)")
    if horizon > 1:
        txt += f"; scaled by the square root of {horizon}, which assumes independent returns and understates risk when volatility clusters"
    add("q_var_headline", "Headline VaR and ES", txt + ".", "info", 0.4, var=v, es=e, confidence=hl, method=base)
    vn, en = cur.get(("normal", hl), (None, None))
    if vn and v == v and ku > 1:
        add("q_var_tails", "Normal VaR understates tail risk", f"Returns have excess kurtosis {ku:.1f} and skewness {sk:+.2f}. At {hl:.0%} the normal-model VaR is {vn:.2%} against {v:.2%} historical "
            f"({(v / vn - 1):+.0%}); the normal expected shortfall is {en:.2%} against {e:.2%}.", "notable" if v / vn > 1.15 else "info", min(1.0, abs(v / vn - 1)), kurtosis=ku, skew=sk)
    for (m, a), b in bts.items():
        rej = b["p_cc"] < 0.05 or b["p_pof"] < 0.05
        if rej:
            why = ("too many exceptions" if b["x"] > b["expected"] else "too few exceptions (the model is too conservative)") if b["p_pof"] < 0.05 else "exceptions are clustered in time"
            add("q_var_backtest", f"{METHODS[m]} {a:.0%} VaR rejected", f"{METHODS[m]} {a:.0%} VaR is rejected by backtesting over {b['N']:,} out-of-sample days: {b['x']} exceptions versus {b['expected']:.1f} expected "
                f"(Kupiec p = {b['p_pof']:.3f}, conditional-coverage p = {b['p_cc']:.3f}): {why}."
                + (f" The average loss on exception days is {b['es_ratio']:.2f}x the model's expected shortfall." if np.isfinite(b["es_ratio"]) else "")
                + (f" Basel traffic light over the last 250 days: {b['light']}." if b["light"] else ""), "warning" if b["light"] == "red" or b["p_pof"] < 0.01 else "notable", 0.7,
                method=m, confidence=a, exceptions=b["x"], expected=b["expected"], p_kupiec=b["p_pof"], p_cc=b["p_cc"])
    ok_models = [(m, a) for (m, a), b in bts.items() if a == hl and b["p_cc"] >= 0.05 and b["p_pof"] >= 0.05]
    if bts:
        add("q_var_summary", "Backtest summary", f"At {hl:.0%}, {len(ok_models)} of {sum(1 for (m, a) in bts if a == hl)} methods pass both coverage and independence tests"
            f"{': ' + ', '.join(METHODS[m] for m, _ in ok_models) if ok_models else ''}. Passing means the data are consistent with the model, not that it is correct.", "info", 0.3)
    wd = np.argsort(r)[:5]
    w20 = np.array([r[i:i + 20].sum() for i in range(len(r) - 19)]) if len(r) >= 20 else np.array([])
    stress_rows = [["worst day", str(dates[i]), float(r[i])] for i in wd]
    if len(w20):
        i = int(np.argmin(w20))
        stress_rows.append(["worst 20-period window (sum)", f"{dates[i]} to {dates[min(i + 19, len(dates) - 1)]}", float(w20[i])])
    add("q_var_stress", "Worst observed periods", f"The worst single period lost {-r[wd[0]]:.2%} ({dates[wd[0]]})" + (f" and the worst 20-period window lost {-w20.min():.2%}" if len(w20) else "") +
        f"; the {hl:.0%} historical VaR is {v:.2%}, so the worst day was {(-r[wd[0]]) / v:.1f}x VaR.", "info", 0.3)
    for i, f in enumerate(facts, 1):
        f.id = f"V{i:03d}"
    charts = {}
    bins = 50
    lo, hi = float(r.min()), float(r.max())
    w = (hi - lo) / bins
    counts = np.bincount(np.minimum(((r - lo) / w).astype(int), bins - 1), minlength=bins)
    lines = [{"x": -v, "label": f"{hl:.0%} VaR"}] + ([{"x": -e, "label": "ES"}] if e == e else [])
    charts[facts[0].id] = ({"kind": "hist", "mn": lo, "w": w, "n": len(r), "xlabel": "return per period", "ylabel": "periods", "lines": lines}, [(i, int(c)) for i, c in enumerate(counts) if c])
    if (base, hl) in bts:
        b = bts[(base, hl)]
        d0 = dates[window:]
        rows = [(d0[i], float(r[window + i]), float(-b["var"][i])) for i in range(len(d0))][-1500:]
        charts[facts[-1].id] = ({"kind": "multiline", "names": ["return", f"-VaR ({METHODS[base]}, {hl:.0%})"], "xlabel": "time", "ylabel": "return"}, rows)
    summary = {"series": series.name, "returns": len(r), "mean per period": mu, "volatility per period": sd, "skewness": sk, "excess kurtosis": ku, "estimation window": window,
               "backtest days": len(r) - window, "horizon": horizon, "position": position if position else "n/a"}
    out = SubReport(kind="var", title=f"Value at Risk: {series.name}", summary=summary, facts=facts, setup_sql=[series.source.source_sql()], warnings=list(series.notes), charts=charts,
                    tables={"VaR and ES": (["method", "confidence", "VaR", "ES"] + (["VaR (amount)", "ES (amount)"] if position else []), rows_now),
                            "Backtest": (["method", "confidence", "days", "exceptions", "expected", "rate", "Kupiec p", "independence p", "conditional p", "ES ratio", "Basel light"], rows_bt),
                            "Worst periods": (["what", "when", "return"], stress_rows)})
    out.provenance = provenance.build([provenance.input_record(series.source, "data", cfg.hash_data)], {"confidence": list(confidence), "methods": methods, "window": window, "horizon": horizon, "position": position},
                                      cfg, None, None, facts, command)
    progress("Done", 100)
    return out
