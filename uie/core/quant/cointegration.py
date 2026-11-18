"""Engle-Granger cointegration for pairs (and a scan over all pairs of a panel), with hedge ratio, spread half-life, current spread
z-score, hedge-ratio stability and a spurious-regression check. MacKinnon p-values; Benjamini-Hochberg across the pairs scanned."""
from __future__ import annotations
import itertools
import math
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from .mackinnon import mackinnon_crit, mackinnon_p
from .series import Panel


def engle_granger(y, x):
    """y on [1, x] by OLS, ADF (no constant) on the residuals, MacKinnon critical values for 2 or more variables."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float).reshape(len(y), -1)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    a = stats.adf(u, "n")
    if a is None:
        return None
    nv = X.shape[1]
    crit = mackinnon_crit(nv, "c", len(u))
    return {"stat": a.stat, "p": mackinnon_p(a.stat, nv, "c"), "crit": crit, "alpha": float(beta[0]), "beta": beta[1:], "resid": u, "lags": a.lags, "n": len(u)}


def half_life(spread):
    s = np.asarray(spread, float)
    ds, lag = np.diff(s), s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    b = np.linalg.lstsq(X, ds, rcond=None)[0][1]
    phi = 1 + b
    return (math.log(0.5) / math.log(phi)) if 0 < phi < 1 else float("inf"), float(phi)


def cointegration(panel: Panel, log: bool = True, max_pairs: int = 80, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    if panel.kind != "price":
        raise ValueError("Cointegration is tested on price levels; this panel holds returns.")
    X = np.log(panel.X) if log else panel.X
    N = X.shape[1]
    names = panel.names
    pairs = list(itertools.combinations(range(N), 2))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    uni = []                                   # unit-root test of each series on its own: cointegration is only meaningful for I(1) series
    for j in range(N):
        a_ = stats.adf(X[:, j], "c")
        uni.append({"stat": a_.stat if a_ else float("nan"), "p": a_.p if a_ else float("nan")})
    stationary = {names[j] for j in range(N) if uni[j]["p"] == uni[j]["p"] and uni[j]["p"] < 0.05}
    res = []
    for k, (i, j) in enumerate(pairs):
        progress(f"Testing {names[i]} / {names[j]}", 10 + int(75 * k / len(pairs)))
        a, b = engle_granger(X[:, i], X[:, j]), engle_granger(X[:, j], X[:, i])
        if a is None or b is None:
            continue
        first, dep, ind = (a, i, j) if a["p"] <= b["p"] else (b, j, i)
        hl, phi = half_life(first["resid"])
        h = len(first["resid"]) // 2
        b1 = np.polyfit(X[:h, ind], X[:h, dep], 1)[0]
        b2 = np.polyfit(X[h:, ind], X[h:, dep], 1)[0]
        sd = first["resid"].std(ddof=1)
        res.append({"i": dep, "j": ind, "stat": first["stat"], "p": first["p"], "crit": first["crit"], "beta": float(first["beta"][0]), "alpha": first["alpha"], "hl": hl, "phi": phi,
                    "z": float((first["resid"][-1] - first["resid"].mean()) / sd) if sd else 0.0, "b1": float(b1), "b2": float(b2),
                    "corr": float(np.corrcoef(X[:, i], X[:, j])[0, 1]), "resid": first["resid"], "n": first["n"], "p_other": max(a["p"], b["p"]),
                    "flag": [names[x] for x in (dep, ind) if names[x] in stationary]})
    if not res:
        raise ValueError("No pair could be tested.")
    # Multiple-testing correction only over pairs where the test is meaningful (both series look I(1)); pairs with a stationary-looking
    # series would otherwise inflate the family with artefactual rejections and make borderline pairs pass.
    fam = [r for r in res if not r["flag"]]
    adj = stats.bh_adjust([min(1.0, 2 * r["p"]) for r in fam]) if fam else []          # x2: both orderings were tried
    for r, pa in zip(fam, adj):
        r["p_adj"] = float(pa)
    for r in res:
        if r["flag"]:
            r["p_adj"] = min(1.0, 2 * r["p"])
    res.sort(key=lambda r: r["p"])
    facts = []
    sql = panel.sql

    def add(type_, title, msg, sev, score, cols, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=cols))

    sig_all = [r for r in res if r["p_adj"] < 0.05]
    sig = [r for r in sig_all if not r["flag"]]
    for r in [x for x in sig_all if x["flag"]][:4]:
        fl = ", ".join(r["flag"])
        add("q_coint_caveat", f"{names[r['i']]} / {names[r['j']]}: not a reliable cointegration result",
            f"The pair {names[r['i']]} / {names[r['j']]} passes the Engle-Granger test (p = {r['p']:.4f}), but {fl} already looks stationary on its own (ADF p = "
            f"{min(uni[names.index(n)]['p'] for n in r['flag']):.4f}). When one series is stationary the residual is stationary whatever the other series does, so the result does not show a shared "
            f"stochastic trend. Check the series individually before trading this spread.", "info", 0.3, [names[r["i"]], names[r["j"]]], flagged=r["flag"], p=r["p"])
    for r in sig[:6]:
        dep, ind = names[r["i"]], names[r["j"]]
        stable = abs(r["b1"] - r["b2"]) / max(abs(r["b1"]), 1e-9) < 0.25
        hl_txt = f"{r['hl']:.0f} periods" if np.isfinite(r["hl"]) else "no mean reversion"
        add("q_coint_pair", f"{dep} / {ind} cointegrated",
            f"{dep} and {ind} are cointegrated (Engle-Granger statistic {r['stat']:.2f}, p = {r['p']:.4f}, {r['p_adj']:.3f} after correcting for {len(res)} pairs x 2 orderings). "
            f"Hedge ratio {r['beta']:.3f} ({'log' if log else 'level'} prices), spread half-life {hl_txt}, current spread {r['z']:+.2f} standard deviations from its mean. "
            f"The hedge ratio is {'stable' if stable else 'NOT stable'} across the two halves of the sample ({r['b1']:.3f} then {r['b2']:.3f}).",
            "notable" if stable else "info", min(1.0, 1 - r["p"]), [dep, ind], beta=r["beta"], p=r["p"], half_life=r["hl"], z=r["z"], stable=bool(stable))
    spurious = [r for r in res if abs(r["corr"]) >= 0.9 and r["p"] > 0.10]
    if stationary:
        add("q_coint_stationary", "Series that look stationary", f"{', '.join(sorted(stationary))} {'looks' if len(stationary) == 1 else 'look'} stationary on {'its' if len(stationary) == 1 else 'their'} own "
            f"(ADF rejects a unit root at 5%). Either they are genuinely mean-reverting (so pair tests involving them are not cointegration tests) or this is one of the ~5% of random walks that "
            f"pass by chance.", "info", 0.3, sorted(stationary), series=sorted(stationary))
    for r in spurious[:3]:
        add("q_coint_spurious", f"{names[r['i']]} / {names[r['j']]} look related but are not cointegrated",
            f"{names[r['i']]} and {names[r['j']]} correlate at {r['corr']:+.2f} in levels, but the regression residual has a unit root (p = {r['p']:.2f}). "
            f"The high level correlation reflects shared trends, not a stable long-run relationship; trading the spread would carry unbounded risk.", "notable", 0.5,
            [names[r["i"]], names[r["j"]]], corr=r["corr"], p=r["p"])
    if not sig:
        add("q_coint_none", "No cointegrated pairs", f"None of the {len(fam)} testable pairs is cointegrated after multiple-testing correction. With {len(fam)} pairs at a 5% level you would "
            f"expect about {0.05 * len(fam):.1f} false positives without correction.", "info", 0.2, names)
    for i, f in enumerate(facts, 1):
        f.id = f"K{i:03d}"
    charts = {}
    if sig:
        r = sig[0]
        u = r["resid"]
        mu, sd = float(u.mean()), float(u.std(ddof=1))
        dts = panel.dates[-len(u):]
        rows = [(dts[t], float(u[t]), mu + 2 * sd, mu - 2 * sd, mu) for t in range(len(u))][-1500:]
        fid = next(f.id for f in facts if f.type == "q_coint_pair")
        charts[fid] = ({"kind": "multiline", "names": ["spread", "+2 sd", "-2 sd", "mean"], "xlabel": "time", "ylabel": f"{names[r['i']]} - {r['beta']:.3f} x {names[r['j']]}"}, rows)
    table = [[f"{names[r['i']]} ~ {names[r['j']]}", r["stat"], r["p"], r["p_adj"], r["beta"], r["hl"] if np.isfinite(r["hl"]) else float("nan"), r["z"], r["corr"], r["b1"], r["b2"],
              ", ".join(r["flag"])] for r in res]
    summary = {"assets": ", ".join(names[:8]) + (" ..." if N > 8 else ""), "observations": X.shape[0], "pairs tested": len(res), "cointegrated (adj p < 0.05, I(1) series)": len(sig), "pairs with a stationary-looking series": len(sig_all) - len(sig),
               "price scale": "log" if log else "level", "test": "Engle-Granger, both orderings, MacKinnon p-values", "multiple testing": "Benjamini-Hochberg (x2 orderings)"}
    out = SubReport(kind="coint", title="Cointegration scan", summary=summary, facts=facts, setup_sql=[panel.source.source_sql()], warnings=list(panel.notes), charts=charts,
                    tables={"Pairs": (["pair (y ~ x)", "EG stat", "p", "adj p", "hedge ratio", "half-life", "spread z", "level corr", "beta 1st half", "beta 2nd half", "stationary-looking series"], table),
                            "Series (ADF)": (["series", "ADF stat", "p", "verdict"], [[names[j], uni[j]["stat"], uni[j]["p"], "looks stationary" if names[j] in stationary else "unit root not rejected"] for j in range(N)])})
    out.provenance = provenance.build([provenance.input_record(panel.source, "data", cfg.hash_data)], {"log": log, "assets": names}, cfg, None, None, facts, command)
    progress("Done", 100)
    return out
