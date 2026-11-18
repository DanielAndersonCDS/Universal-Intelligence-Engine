"""Distribution shape: skewness, kurtosis, Jarque-Bera, and the best-fitting simple distribution."""
import numpy as np
from ..facts import Fact
from ..util import qident
from .. import stats

BINS = 50


def detect(ctx):
    cols = [c for c in ctx.measures("distributions") if c.distinct > 20 and ctx.profile.rows - c.nulls >= 200][:30]
    if not cols:
        return []
    rows = ctx.sample_rows([c.name for c in cols])
    X = np.array(rows, dtype=float).reshape(len(rows), len(cols))
    out = []
    for j, c in enumerate(cols):
        x = X[:, j]
        x = x[np.isfinite(x)]
        jb = stats.jarque_bera(x)
        if not jb or jb["p"] > 0.001 or (abs(jb["skew"]) < 1 and abs(jb["kurtosis"]) < 3):
            continue
        fits = stats.fit_distributions(x)
        nrm = next((f for f in fits if f["dist"] == "normal"), None)
        if not fits or not nrm or nrm["ks"] < 0.06 or fits[0]["dist"] == "normal" or fits[0]["ks"] > nrm["ks"] - 0.03:
            continue
        best = fits[0]
        q, rel = qident(c.name), ctx.rel
        full = ctx.con.execute(f"SELECT skewness({q}), kurtosis({q}), count({q}) FROM {rel}").fetchone()
        log_view = best["dist"] == "lognormal" and c.min > 0           # a lognormal looks like a normal on a log axis, and is readable there
        lo, hi = (float(np.log(c.min)), float(np.log(c.max))) if log_view else (c.min, c.max)
        w = (hi - lo) / BINS
        ex = f"ln({q})" if log_view else q
        fit = {"dist": "normal", "params": best["params"]} if log_view else {"dist": best["dist"], "params": best["params"]}
        out.append(Fact(
            type="distribution_fit", title=f"Distribution of {c.name}", columns=[c.name],
            metrics={"column": c.name, "skew": float(full[0] or jb["skew"]), "kurtosis": float(full[1] if full[1] is not None else jb["kurtosis"]),
                     "best": best["dist"], "best_ks": best["ks"], "normal_ks": nrm["ks"], "n": int(full[2]),
                     "ranking": [[f["dist"], f["ks"]] for f in fits], "params": best["params"]},
            severity="info", score=min(0.6, nrm["ks"]),
            evidence_sql=f"SELECT skewness({q}) AS skewness, kurtosis({q}) AS excess_kurtosis, avg({q}) AS mean, median({q}) AS median, stddev_samp({q}) AS std, count({q}) AS n FROM {rel}",
            chart={"kind": "hist", "mn": lo, "w": w, "n": int(full[2]), "fit": fit,
                   "xlabel": f"ln({c.name})" if log_view else c.name, "ylabel": "rows",
                   "sql": f"SELECT least(floor(({ex} - ({lo!r})) / {w!r}), {BINS - 1})::INTEGER AS x, count(*) AS y FROM {rel} WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 1"}))
    out.sort(key=lambda f: -f.score)
    return out[:8]
