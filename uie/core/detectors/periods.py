"""Period-over-period variance breakdowns and time-series gap detection."""
from __future__ import annotations
import re
from ..facts import Fact
from ..util import qident

_HINT = re.compile(r"rev|sales|amount|amt|cost|spend|expense|income|profit|qty|quantity|units|volume|total|balance|payment", re.I)
_SKIP = re.compile(r"age|rate|ratio|pct|percent|discount|score|price|id$", re.I)
BASIS = {"month": "month over month", "week": "week over week", "day": "day over day", "year": "year over year"}
YEAR = {"month": "12 MONTH", "week": "52 WEEK", "day": "364 DAY"}


def detect(ctx):
    return _variance(ctx) + _gaps(ctx)


def _measures(ctx):
    amt = [n for n in ctx.roles("amount")]
    ms = [c for c in ctx.measures("trends") if c.min is not None and c.min >= 0 and ctx.sem.role(c.name) != "price"]
    hinted = [c for c in ms if _HINT.search(c.name)]
    pick = [ctx.col(n) for n in amt if ctx.col(n)] or hinted or [c for c in ms if not _SKIP.search(c.name) and c.distinct > 10][:2]
    return [c for c in pick if c.kind == "numeric"][:3]


def _label(d, unit):
    s = str(d)
    return s[:7] if unit == "month" else s


def _variance(ctx):
    t, cfg, rel = ctx.time_col, ctx.cfg, ctx.rel
    tc = ctx.col(t) if t else None
    if not tc or tc.min is None or tc.max is None:
        return []
    span = (tc.max - tc.min).total_seconds() / 86400
    if span < 45:
        return []
    unit = "month" if span >= 90 else "week" if span >= 21 else "day"
    qt = qident(t)
    U = unit.upper()
    mx, b_last, b_next = ctx.con.execute(
        f"SELECT max({qt}), CAST(date_trunc('{unit}', max({qt})) AS DATE), CAST(date_trunc('{unit}', max({qt})) + INTERVAL 1 {U} AS DATE) FROM {rel}").fetchone()
    mx_d = mx.date() if hasattr(mx, "date") else mx
    from datetime import timedelta
    cur_complete = mx_d >= b_next - timedelta(days=1)
    cur = ctx.con.execute(f"SELECT CAST(DATE '{b_last}' - INTERVAL {'0' if cur_complete else '1'} {U} AS DATE)").fetchone()[0]
    prior = ctx.con.execute(f"SELECT CAST(DATE '{cur}' - INTERVAL 1 {U} AS DATE)").fetchone()[0]
    yoy = ctx.con.execute(f"SELECT CAST(DATE '{cur}' - INTERVAL {YEAR[unit]} AS DATE)").fetchone()[0]
    bk = f"CAST(date_trunc('{unit}', {qt}) AS DATE)"
    dims = sorted([c for c in ctx.profile.columns if c.kind == "text" and 2 <= c.distinct <= 50 and not ctx.is_id_like(c)
                   and ctx.sem.role(c.name) not in ("ignore", "id") and c.nulls < 0.5 * ctx.profile.rows], key=lambda c: c.nulls)[:3]
    out = []
    for m in _measures(ctx):
        qm = qident(m.name)
        for basis_unit, ref in ((unit, prior), ("year", yoy)):
            if basis_unit == "year" and ref >= prior:
                continue
            tot = ctx.con.execute(
                f"SELECT sum({qm}) FILTER (WHERE {bk} = DATE '{cur}'), sum({qm}) FILTER (WHERE {bk} = DATE '{ref}'), "
                f"count(*) FILTER (WHERE {bk} = DATE '{cur}'), count(*) FILTER (WHERE {bk} = DATE '{ref}') FROM {rel}").fetchone()
            c_v, p_v, c_n, p_n = tot
            if c_v is None or p_v is None or not p_v or c_n < cfg.pop_min_rows or p_n < cfg.pop_min_rows:
                continue
            delta = float(c_v) - float(p_v)
            change = delta / abs(float(p_v))
            if abs(change) < cfg.pop_min_change:
                continue
            best = None
            for d in dims:
                qd = qident(d.name)
                rs = ctx.con.execute(
                    f"SELECT coalesce(CAST({qd} AS VARCHAR), '(missing)'), coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{cur}'), 0), "
                    f"coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{ref}'), 0) FROM {rel} WHERE {bk} IN (DATE '{cur}', DATE '{ref}') GROUP BY 1").fetchall()
                contrib = sorted(((g, float(a) - float(b)) for g, a, b in rs), key=lambda x: -x[1] * (1 if delta > 0 else -1))
                top = [x for x in contrib if x[1] * delta > 0][:3]
                if len(rs) < 2 or not top:
                    continue
                expl = sum(x[1] for x in top) / delta
                if best is None or expl > best[0]:
                    best = (expl, d, top)
            if not best:
                continue
            _, d, top = best
            qd = qident(d.name)
            ev = (f"SELECT coalesce(CAST({qd} AS VARCHAR), '(missing)') AS {qident(d.name)}, sum({qm}) FILTER (WHERE {bk} = DATE '{cur}') AS current, "
                  f"sum({qm}) FILTER (WHERE {bk} = DATE '{ref}') AS prior, coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{cur}'), 0) - "
                  f"coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{ref}'), 0) AS delta FROM {rel} WHERE {bk} IN (DATE '{cur}', DATE '{ref}') GROUP BY 1 ORDER BY 4 DESC")
            out.append(Fact(type="variance_driver", title=f"{m.name} {BASIS[basis_unit]}", columns=[m.name, d.name, t],
                            metrics={"measure": m.name, "basis": BASIS[basis_unit], "direction": "rose" if delta > 0 else "fell",
                                     "pct": change, "prior": float(p_v), "current": float(c_v), "prior_label": _label(ref, unit),
                                     "current_label": _label(cur, unit), "dimension": d.name,
                                     "top": [{"group": g, "delta": v, "share": v / delta} for g, v in top]},
                            severity="notable" if abs(change) >= 0.15 else "info", score=min(1.0, abs(change)),
                            evidence_sql=ev,
                            chart={"kind": "bar", "xlabel": d.name, "ylabel": f"change in {m.name}",
                                   "sql": f"SELECT g AS x, delta AS y FROM (SELECT coalesce(CAST({qd} AS VARCHAR), '(missing)') AS g, coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{cur}'), 0) - coalesce(sum({qm}) FILTER (WHERE {bk} = DATE '{ref}'), 0) AS delta FROM {rel} WHERE {bk} IN (DATE '{cur}', DATE '{ref}') GROUP BY 1) ORDER BY abs(delta) DESC"}))
    return out


def _gaps(ctx):
    t, cfg, rel = ctx.time_col, ctx.cfg, ctx.rel
    if not t or ctx.profile.rows < 30:
        return []
    qt = qident(t)
    base = (f"SELECT d, lead(d) OVER (ORDER BY d) AS nxt FROM (SELECT DISTINCT CAST({qt} AS DATE) AS d FROM {rel} WHERE {qt} IS NOT NULL)")
    row = ctx.con.execute(f"WITH g AS ({base}) SELECT median(date_diff('day', d, nxt)), count(*) FROM g WHERE nxt IS NOT NULL").fetchone()
    typical, n = row
    if typical is None or n < 10:
        return []
    typical = float(typical)
    thr = max(cfg.gap_factor * typical, cfg.gap_min_days)
    gaps = ctx.con.execute(
        f"WITH g AS ({base}) SELECT d, nxt, date_diff('day', d, nxt) AS gap FROM g WHERE nxt IS NOT NULL AND date_diff('day', d, nxt) > {thr!r} ORDER BY gap DESC").fetchall()
    if not gaps:
        return []
    d0, d1, longest = gaps[0]
    return [Fact(type="time_gap", title=f"Gaps in {t}", columns=[t],
                 metrics={"column": t, "gaps": len(gaps), "typical_days": typical, "longest_days": float(longest),
                          "longest_start": str(d0), "longest_end": str(d1), "threshold_days": thr},
                 severity="notable" if longest >= 4 * thr or len(gaps) >= 5 else "info", score=min(1.0, longest / (10 * thr)),
                 evidence_sql=f"WITH g AS ({base}) SELECT d AS last_date_before_gap, nxt AS next_date, date_diff('day', d, nxt) AS gap_days FROM g "
                              f"WHERE nxt IS NOT NULL AND date_diff('day', d, nxt) > {thr!r} ORDER BY gap_days DESC LIMIT 100")]
