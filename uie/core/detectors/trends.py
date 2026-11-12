import math
import numpy as np
from ..facts import Fact
from ..util import qident, to_ts, ts_str
from .. import stats

UNITS = [("hour", 1 / 24), ("day", 1), ("week", 7), ("month", 30.44), ("quarter", 91.3), ("year", 365.25)]


def pick_unit(days):
    for name, size in UNITS:
        if days / size <= 400:
            return name
    return "year"


def detect(ctx):
    t = ctx.time_col
    if not t:
        return []
    tc = ctx.profile.col(t)
    if not tc or tc.min is None or tc.max is None:
        return []
    days = (tc.max - tc.min).total_seconds() / 86400
    unit = pick_unit(days)
    cols = [c for c in ctx.measures("trends") if c.name != t][:12]
    qt, rel = qident(t), ctx.rel
    sel = ", ".join(["count(*) AS n"] + [f"avg({qident(c.name)})" for c in cols])
    rows = ctx.con.execute(
        f"SELECT date_trunc('{unit}', {qt}) AS bucket, {sel} FROM {rel} WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall()
    if len(rows) < 8:
        return []
    ts = np.array([to_ts(r[0]) for r in rows])
    out = []
    vol = np.array([float(r[1]) for r in rows])
    out += _analyse(ts[1:-1], vol[1:-1], f"Record volume (per {unit})", "records", unit,
                    ctx.cfg, f"SELECT date_trunc('{unit}', {qt}) AS x, count(*) AS y FROM {rel} WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1")
    for j, c in enumerate(cols):
        y = np.array([np.nan if r[2 + j] is None else float(r[2 + j]) for r in rows])
        out += _analyse(ts, y, f"The average of `{c.name}`", c.name, unit, ctx.cfg,
                        f"SELECT date_trunc('{unit}', {qt}) AS x, avg({qident(c.name)}) AS y FROM {rel} WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1")
    return out


def _analyse(ts, y, label, col, unit, cfg, sql):
    m = np.isfinite(y)
    yy, t = y[m], ts[m]
    n = len(yy)
    if n < 8:
        return []
    x = np.arange(n)
    slope, icpt = np.polyfit(x, yy, 1)
    ss_res = float(((yy - (slope * x + icpt)) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    if ss_tot <= 0:
        return []
    r2 = 1 - ss_res / ss_tot
    k = max(2, n // 3)
    start, end = float(yy[:k].mean()), float(yy[-k:].mean())
    change = (end - start) / abs(start) if start else 0.0
    period = f"{ts_str(t[0])}..{ts_str(t[-1])}"
    out, trend_ok = [], False
    chart = {"kind": "line", "sql": sql, "xlabel": f"time ({unit})", "ylabel": col}
    # Stationarity awareness: a linear fit on a random walk "finds" trends constantly. Test for a unit root first
    # (ADF with constant + trend); if it is not rejected, require a significant drift in the differences instead.
    adf = stats.adf(yy, "ct") if n >= cfg.adf_min_points else None
    nonstat = adf is not None and not adf.reject5
    drift_t = stats.nw_mean_t(np.diff(yy)) if adf is not None else None
    if r2 >= 0.5 and abs(change) >= 0.05:
        if nonstat and abs(drift_t) < cfg.drift_t_min:
            out.append(Fact(type="trend_caveat", title=f"Apparent trend in {col} is not significant", columns=[col],
                            metrics={"label": label, "direction": "rose" if change > 0 else "fell", "pct": abs(change), "r2": r2,
                                     "adf": adf.stat, "drift_t": drift_t},
                            severity="info", score=0.2, evidence_sql=sql, chart=chart))
        else:
            trend_ok = True
            m = {"label": label, "direction": "rose" if change > 0 else "fell", "pct": abs(change),
                 "start": start, "end": end, "r2": r2, "period": period, "unit": unit, "points": n}
            if adf is not None:
                m.update(regime="unit root with significant drift" if nonstat else "trend-stationary", adf=adf.stat, drift_t=drift_t)
            out.append(Fact(type="trend", title=f"Trend in {col}", columns=[col], metrics=m,
                            severity="notable" if abs(change) >= 0.25 else "info",
                            score=r2 * min(abs(change), 2) / 2, evidence_sql=sql, chart=chart))
    if n >= 12:
        cs, cq = np.cumsum(yy), np.cumsum(yy ** 2)
        best = None
        for s in range(4, n - 3):
            sl, ql = cs[s - 1], cq[s - 1]
            sr, qr = cs[-1] - sl, cq[-1] - ql
            sse = (ql - sl * sl / s) + (qr - sr * sr / (n - s))
            if best is None or sse < best[0]:
                best = (sse, s)
        sse, s = best
        gain = 1 - sse / ss_tot
        m1, m2 = float(yy[:s].mean()), float(yy[s:].mean())
        pooled = math.sqrt(max(sse, 1e-12) / (n - 2))
        effect = abs(m2 - m1) / pooled
        rel = abs(m2 - m1) / abs(m1) if m1 else 0.0
        jump_ok = True
        if nonstat:   # a random walk wanders; only accept a level shift that shows up as a genuine jump in the differences
            z = stats.robust_z(np.diff(yy))
            jump_ok = float(z[max(0, s - 3):s + 2].max()) >= cfg.jump_z_min
        if jump_ok and gain >= 0.4 and effect >= 2.0 and rel >= 0.10 and sse < 0.7 * ss_res and not (trend_ok and r2 >= 0.9):
            out.append(Fact(type="changepoint", title=f"Level shift in {col}", columns=[col],
                            metrics={"label": label, "when": ts_str(t[s]), "before": m1, "after": m2, "pct": rel,
                                     "direction": "rose" if m2 > m1 else "fell", "effect_size": effect},
                            severity="notable", score=min(1, gain * min(rel, 2) / 2 + 0.2), evidence_sql=sql,
                            chart={**chart, "marker_ts": float(t[s])}))
    return out
