"""Returns-based analysis for price series: volatility, volatility regime shifts, fat tails, and return correlations.
Price columns come from the semantics file (role: price) or exact name matches (close, adj_close, price, nav ...)."""
from __future__ import annotations
import math
from ..facts import Fact
from ..semantics import _lit
from ..util import qident


def _unit(ctx, t):
    row = ctx.con.execute(
        f"SELECT median(g) FROM (SELECT date_diff('day', d, lead(d) OVER (ORDER BY d)) AS g FROM "
        f"(SELECT DISTINCT CAST({qident(t)} AS DATE) AS d FROM {ctx.rel})) WHERE g IS NOT NULL").fetchone()
    g = row[0] if row and row[0] is not None else 1
    return "day" if g <= 1.5 else "week" if g <= 10 else "month"


def _bucket_sql(rel, t, unit, p, ent=None, flt=""):
    e = f"{qident(ent)} AS ent, " if ent else ""
    grp = "1, 2" if ent else "1"
    return (f"SELECT {e}CAST(date_trunc('{unit}', {qident(t)}) AS DATE) AS b, arg_max({qident(p)}, {qident(t)}) AS p FROM {rel} "
            f"WHERE {qident(p)} > 0 AND {qident(t)} IS NOT NULL {flt} GROUP BY {grp}")


def _ret_cte(bsql, ent):
    part = "PARTITION BY ent " if ent else ""
    return (f"WITH b AS ({bsql}), r AS (SELECT {'ent, ' if ent else ''}b, p, ln(p / lag(p) OVER ({part}ORDER BY b)) AS r, "
            f"ntile(3) OVER ({part}ORDER BY b) AS tile FROM b)")


def detect(ctx):
    t, cfg = ctx.time_col, ctx.cfg
    prices = [c for c in (ctx.col(n) for n in ctx.roles("price")) if c and c.kind == "numeric" and (c.min or 0) > 0]
    if not t or not prices:
        return []
    if ctx.group and ctx.entity == ctx.group["column"]:
        return []                          # the pooled pass already analysed every entity
    ent = ctx.entity if (ctx.entity and not ctx.group and ctx.col(ctx.entity)) else None
    unit = _unit(ctx, t)
    out, rel = [], ctx.rel
    ents = []
    if ent:
        ents = [r[0] for r in ctx.con.execute(
            f"SELECT {qident(ent)} FROM {rel} WHERE {qident(ent)} IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {cfg.returns_max_entities}").fetchall()]
        if not ents:
            ent = None
    flt = f"AND {qident(ent)} IN ({', '.join(_lit(v) for v in ents)})" if ent else ""
    for pc in prices:
        bsql = _bucket_sql(rel, t, unit, pc.name, ent, flt)
        cte = _ret_cte(bsql, ent)
        sel_e = "ent, " if ent else ""
        rows = ctx.con.execute(
            f"{cte} SELECT {sel_e}count(r), stddev_samp(r), avg(r), kurtosis(r), min(r), max(r), min(b), max(b), "
            f"stddev_samp(r) FILTER (WHERE tile < 3), stddev_samp(r) FILTER (WHERE tile = 3), count(r) FILTER (WHERE tile = 3) "
            f"FROM r WHERE r IS NOT NULL GROUP BY {'ent' if ent else '1=1'}").fetchall()
        for row in rows:
            val = row[0] if ent else None
            n, sd, mu, kurt, lo, hi, b0, b1, sd_early, sd_late, n_late = row[1 if ent else 0:][:11]
            if not n or n < 30 or not sd:
                continue
            years = max((b1 - b0).days / 365.25, 1e-9)
            ppy = n / years
            ann = sd * math.sqrt(ppy)
            who = f"`{pc.name}`" + (f" for {ent} = {val}" if ent else "")
            esel = f"AND ent = {_lit(val)}" if ent else ""
            ev = f"{cte} SELECT count(r) AS n, stddev_samp(r) AS std_log_return, kurtosis(r) AS excess_kurtosis, min(r), max(r) FROM r WHERE r IS NOT NULL {esel}"
            chart = {"kind": "line", "xlabel": f"time ({unit})", "ylabel": pc.name,
                     "sql": f"{cte.split(', r AS')[0]} SELECT b, p FROM b {('WHERE ent = ' + _lit(val)) if ent else ''} ORDER BY b"}
            out.append(Fact(type="volatility", title=f"Volatility of {pc.name}" + (f" ({val})" if ent else ""), columns=[pc.name] + ([ent] if ent else []),
                            metrics={"label": who, "column": pc.name, "entity": None if val is None else str(val), "n": int(n),
                                     "ann_vol": ann, "ppy": ppy, "worst": float(lo), "best": float(hi), "kurtosis": float(kurt or 0.0)},
                            severity="info", score=min(0.5, ann), evidence_sql=ev, chart=chart))
            if sd_early and sd_late and n_late and n_late >= 20:
                ratio = sd_late / sd_early
                if ratio >= cfg.vol_shift_ratio or ratio <= 1 / cfg.vol_shift_ratio:
                    out.append(Fact(type="vol_shift", title=f"Volatility regime shift in {pc.name}" + (f" ({val})" if ent else ""),
                                    columns=[pc.name] + ([ent] if ent else []),
                                    metrics={"label": who, "ratio": max(ratio, 1 / ratio), "direction": "rose" if ratio > 1 else "fell",
                                             "early_vol": sd_early * math.sqrt(ppy), "late_vol": sd_late * math.sqrt(ppy)},
                                    severity="notable", score=min(1.0, abs(math.log(ratio))), evidence_sql=ev, chart=chart))
    if not ctx.group:
        out += _correlations(ctx, prices, ent, ents, unit)
    return out


def _correlations(ctx, prices, ent, ents, unit):
    cfg, rel, t = ctx.cfg, ctx.rel, ctx.time_col
    qt, bt = qident(t), f"CAST(date_trunc('{unit}', {qident(t)}) AS DATE)"
    if len(prices) >= 2:
        series = [(pc.name, f"arg_max({qident(pc.name)}, {qt})") for pc in prices[:10]]
        flt = ""
    elif ent and len(ents) >= 2:
        pc = prices[0]
        series = [(f"{ent}={v}", f"arg_max({qident(pc.name)}, {qt}) FILTER (WHERE {qident(ent)} = {_lit(v)})") for v in ents[:12]]
        flt = f"AND {qident(pc.name)} > 0"
    else:
        return []
    names = [n for n, _ in series]
    w = (f"WITH w AS (SELECT {bt} AS b, " + ", ".join(f"{e} AS s{i}" for i, (_, e) in enumerate(series)) +
         f" FROM {rel} WHERE {qt} IS NOT NULL {flt} GROUP BY 1), "
         "r AS (SELECT b, " + ", ".join(f"s{i}, ln(s{i} / lag(s{i}) OVER (ORDER BY b)) AS r{i}" for i in range(len(series))) + " FROM w)")
    pairs = [(i, j) for i in range(len(series)) for j in range(i + 1, len(series))]
    row = ctx.con.execute(w + " SELECT " + ", ".join(
        f"corr(s{i}, s{j}), corr(r{i}, r{j}), count_if(r{i} IS NOT NULL AND r{j} IS NOT NULL)" for i, j in pairs) + " FROM r").fetchone()
    out, spur, ret = [], [], []
    for k, (i, j) in enumerate(pairs):
        lvl, rr, n = row[3 * k], row[3 * k + 1], row[3 * k + 2]
        if lvl is None or rr is None or n < 30:
            continue
        ev = w + f" SELECT corr(s{i}, s{j}) AS level_r, corr(r{i}, r{j}) AS return_r, count_if(r{i} IS NOT NULL AND r{j} IS NOT NULL) AS n FROM r"
        m = {"a": names[i], "b": names[j], "level_r": float(lvl), "return_r": float(rr), "r": float(rr), "n": int(n)}
        cols = [prices[0].name] if ent else [names[i], names[j]]
        if abs(lvl) >= cfg.spurious_level_r and abs(rr) < cfg.spurious_return_r:
            spur.append(Fact(type="spurious_correlation", title=f"{names[i]} vs {names[j]} (levels vs returns)", columns=cols, metrics=m,
                             severity="notable", score=abs(lvl) - abs(rr), evidence_sql=ev))
        elif abs(rr) >= cfg.return_corr_min:
            ret.append(Fact(type="return_correlation", title=f"{names[i]} vs {names[j]} (returns)", columns=cols, metrics=m,
                            severity="notable" if abs(rr) >= 0.8 else "info", score=abs(rr), evidence_sql=ev))
    spur.sort(key=lambda f: -f.score)
    ret.sort(key=lambda f: -f.score)
    return spur[:8] + ret[:8]
