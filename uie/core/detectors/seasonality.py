"""Repeating cycles: autocorrelation of the (detrended) bucketed series. Reports the cycle length in buckets and how much variation it explains."""
import numpy as np
from ..facts import Fact
from ..util import qident
from .. import stats
from .returns import _unit


def _series_multi(ctx, unit, exprs, t):
    """One query for every series: returns (bucket count, list of arrays or None per expression)."""
    qt = qident(t)
    rows = ctx.con.execute(f"SELECT CAST(date_trunc('{unit}', {qt}) AS DATE) AS b, {', '.join(exprs)} FROM {ctx.rel} WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall()
    if len(rows) < 24:
        return []
    step = {"day": 1, "week": 7}.get(unit)
    b0 = rows[0][0]
    idx = np.array([(r[0] - b0).days // step if step else (r[0].year - b0.year) * 12 + r[0].month - b0.month for r in rows])
    n = int(idx[-1]) + 1
    out = []
    for j in range(len(exprs)):
        vals = [(i, float(r[1 + j])) for i, r in zip(idx, rows) if r[1 + j] is not None]
        if len(vals) < 0.9 * n or len(vals) < 24:        # too many empty buckets to read lags meaningfully
            out.append(None)
            continue
        out.append(np.interp(np.arange(n), [v[0] for v in vals], [v[1] for v in vals]))
    return out


def detect(ctx):
    t, cfg = ctx.time_col, ctx.cfg
    if not t or ctx.profile.rows < 100:
        return []
    unit = _unit(ctx, t)
    ms = [c for c in ctx.measures("seasonality") if c.distinct > 10][:40]
    specs = [("Record volume", "count(*)", None)] + [(f"The average of `{c.name}`", f"avg({qident(c.name)})", c) for c in ms]
    series = _series_multi(ctx, unit, [e for _, e, _ in specs], t)
    out = []
    for (label, expr, c), y in zip(specs, series):
        if y is None or len(y) < 24 or np.ptp(y) == 0:
            continue
        n = len(y)
        a = stats.adf(y, "ct")
        if a is not None and not a.reject5:
            x = np.diff(y)
        else:
            k = np.arange(n)
            x = y - np.polyval(np.polyfit(k, y, 1), k)
        if x.std() == 0:
            continue
        maxlag = min(len(x) // 3, 60)
        ac = stats.acf(x, maxlag)
        thr = max(0.25, 2.5 / np.sqrt(len(x)))
        peaks = [l for l in range(2, maxlag) if ac[l] >= thr and ac[l] >= ac[l - 1] and ac[l] >= ac[l + 1]]
        if not peaks:
            continue
        best = max(ac[l] for l in peaks)
        period = min(l for l in peaks if ac[l] >= 0.8 * best)
        prof = np.array([x[i::period].mean() for i in range(period)])
        fitted = prof[np.arange(len(x)) % period]
        strength = 1 - float(((x - fitted) ** 2).sum() / ((x - x.mean()) ** 2).sum())
        if strength < 0.15:
            continue
        sql = f"SELECT CAST(date_trunc('{unit}', {qident(t)}) AS DATE) AS b, {expr} AS v FROM {ctx.rel} WHERE {qident(t)} IS NOT NULL GROUP BY 1 ORDER BY 1"
        out.append(Fact(type="seasonality", title=f"Cycle in {c.name if c else 'record volume'}", columns=[c.name] if c else [t],
                        metrics={"label": label, "unit": unit, "period": int(period), "acf": float(ac[period]), "strength": strength, "n": int(len(y)),
                                 "differenced": bool(a is not None and not a.reject5)},
                        severity="notable" if strength >= 0.5 else "info", score=min(1.0, strength), evidence_sql=sql,
                        chart={"kind": "acf", "maxlag": maxlag, "xlabel": f"lag ({unit}s)", "ylabel": "autocorrelation", "sql": sql}))
    out.sort(key=lambda f: -f.score)
    return out[:4]
