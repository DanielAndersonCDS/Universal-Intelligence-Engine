import math
from ..facts import Fact
from ..util import qident

BINS = 60


def detect(ctx):
    T = ctx.cfg.outlier_t   # robust z-score threshold (median / MAD)
    cols = [c for c in ctx.measures("outliers") if c.distinct > 10 and c.median is not None][:40]
    if not cols:
        return []
    rel = ctx.rel
    # Strictly positive, right-skewed columns (amounts, prices, durations) are tested on a log scale; a z-score on the
    # raw values would flag a large share of ordinary values as outliers.
    spec = []
    for c in cols:
        log = bool(c.min and c.min > 0 and c.mean and c.median and c.mean / c.median >= 1.5)
        q = qident(c.name)
        ex = f"ln({q})" if log else q
        med = math.log(c.median) if log else c.median
        spec.append((c, log, ex, med))
    mads = ctx.con.execute("SELECT " + ", ".join(f"approx_quantile(abs({ex} - ({med!r})), 0.5)" for c, log, ex, med in spec) + f" FROM {rel}").fetchone()
    sel = []
    for (c, log, ex, med), m in zip(spec, mads):
        s = 1.4826 * float(m or 0)
        if s <= 0 and not log and c.q75 is not None and c.q75 > c.q25:
            s = (c.q75 - c.q25) / 1.349
        if s > 0:
            sel.append((c, log, ex, med, s))
    if not sel:
        return []
    exprs = []
    for c, log, ex, med, s in sel:
        q = qident(c.name)
        cond = f"abs({ex} - ({med!r})) / {s!r} > {T}"
        exprs += [f"count(*) FILTER (WHERE {cond})", f"count({q})",
                  f"min({q}) FILTER (WHERE {cond} AND {ex} < ({med!r}))",
                  f"max({q}) FILTER (WHERE {cond} AND {ex} > ({med!r}))"]
    row = ctx.con.execute(f"SELECT {', '.join(exprs)} FROM {rel}").fetchone()
    out = []
    for i, (c, log, ex, med, s) in enumerate(sel):
        cnt, nn, lo, hi = row[i * 4:i * 4 + 4]
        if not cnt:
            continue
        share = cnt / max(nn, 1)
        q = qident(c.name)
        w = (c.max - c.min) / BINS
        lines = [math.exp(med - T * s), math.exp(med + T * s)] if log else [med - T * s, med + T * s]
        out.append(Fact(
            type="outliers", title=f"Outliers in {c.name}", columns=[c.name],
            metrics={"column": c.name, "count": int(cnt), "share": share, "threshold": T,
                     "low": None if lo is None else float(lo), "high": None if hi is None else float(hi),
                     "median": c.median, "log_scale": log},
            severity="notable" if share >= 0.001 else "info", score=min(1.0, share * 20),
            evidence_sql=f"SELECT * FROM {rel} WHERE abs({ex} - ({med!r})) / {s!r} > {T} ORDER BY abs({ex} - ({med!r})) DESC LIMIT 100",
            chart={"kind": "bins", "mn": c.min, "w": w, "lines": lines,
                   "xlabel": c.name, "ylabel": "log10(count + 1)",
                   "sql": f"SELECT least(floor(({q} - ({c.min!r})) / {w!r}), {BINS - 1})::INTEGER AS x, count(*) AS y "
                          f"FROM {rel} WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 1"}))
    return out
