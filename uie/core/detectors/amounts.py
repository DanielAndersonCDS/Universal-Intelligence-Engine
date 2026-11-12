"""Amount-pattern checks: Benford first-digit conformity, round-number excess, and bunching just below thresholds."""
from __future__ import annotations
import math
from ..facts import Fact
from ..util import qident, fmt_num
from .. import stats


def _amount_cols(ctx):
    declared = [c for c in (ctx.col(n) for n in ctx.roles("amount")) if c and c.kind == "numeric" and not ctx.sem.excluded(c.name, "all")]
    return declared


def detect(ctx):
    out = []
    declared = _amount_cols(ctx)
    for c in declared:
        out += _round(ctx, c) + _bunching(ctx, c)
    cands = {c.name: (c, True) for c in declared if not ctx.sem.excluded(c.name, "benford")}
    for c in ctx.measures("benford"):
        if c.name not in cands and c.distinct >= max(500, 0.05 * (ctx.profile.rows - c.nulls)) and len(cands) < 10:
            cands[c.name] = (c, False)
    for c, is_declared in cands.values():
        out += _benford(ctx, c, is_declared)
    return out


def _benford(ctx, c, declared):
    q, rel, cfg = qident(c.name), ctx.rel, ctx.cfg
    # Benford only makes sense for values spanning several orders of magnitude
    r = ctx.con.execute(f"SELECT approx_quantile(abs({q}), 0.95), approx_quantile(abs({q}), 0.05) FILTER (WHERE {q} <> 0) FROM {rel}").fetchone()
    if not r[0] or not r[1] or r[0] / r[1] < (20 if declared else 100):
        return []
    digit = f"least(9, greatest(1, CAST(floor(abs({q}) / pow(10, floor(log10(abs({q}))))) AS INTEGER)))"
    sql = f"SELECT {digit} AS x, count(*) AS y FROM {rel} WHERE {q} IS NOT NULL AND {q} <> 0 GROUP BY 1 ORDER BY 1"
    rows = dict(ctx.con.execute(sql).fetchall())
    counts = [rows.get(d, 0) for d in range(1, 10)]
    b = stats.benford_stats(counts)
    if not b or b["n"] < cfg.benford_min_n:
        return []
    need = cfg.benford_mad_info if declared else cfg.benford_mad_notable
    if b["mad"] <= need:
        return []
    label = ("nonconformity" if b["mad"] > cfg.benford_mad_notable else "marginal conformity") + " by Nigrini's MAD thresholds"
    return [Fact(type="benford", title=f"First-digit distribution of {c.name}", columns=[c.name],
                 metrics={**b, "column": c.name, "conformity": label, "declared": declared},
                 severity="notable" if b["mad"] > cfg.benford_mad_notable else "info", score=min(1.0, b["mad"] * 20),
                 evidence_sql=sql, chart={"kind": "bar", "xlabel": "first digit", "ylabel": "rows", "sql": sql})]


def _round(ctx, c):
    q, rel, cfg = qident(c.name), ctx.rel, ctx.cfg
    r = ctx.con.execute(
        f"SELECT count_if(abs({q}) >= 1000), count_if(abs({q}) >= 1000 AND abs({q}) % 1000 = 0), count_if(abs({q}) >= 100), "
        f"count_if(abs({q}) >= 100 AND abs({q}) % 100 = 0), count_if({q} IS NOT NULL AND abs({q} - round({q})) > 1e-9), count({q}) FROM {rel}").fetchone()
    n1000, m1000, n100, m100, nonint, nn = [int(v or 0) for v in r]
    if nn < 200:
        return []
    q_ni = nonint / nn
    for mod, n, m in ((1000, n1000, m1000), (100, n100, m100)):
        if n < 200:
            continue
        share, exp = m / n, q_ni / (mod * 100) + (1 - q_ni) / mod
        if m >= cfg.round_min_count and share >= cfg.round_min_share and share >= 5 * exp:
            return [Fact(type="round_numbers", title=f"Round amounts in {c.name}", columns=[c.name],
                         metrics={"column": c.name, "modulus": mod, "count": m, "share": share, "expected": exp, "n": n},
                         severity="notable", score=min(1.0, share * 3),
                         evidence_sql=f"SELECT {q} AS amount, count(*) AS rows FROM {rel} WHERE abs({q}) >= {mod} AND abs({q}) % {mod} = 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 100")]
    return []


def _candidates(ctx, c):
    declared = [float(v) for v in (ctx.sem.get("thresholds") or [])]
    auto = []
    if c.max and c.median is not None and c.max > 0:
        k = math.floor(math.log10(max(c.median, 10)))
        for e in range(k, int(math.log10(c.max)) + 1):
            for m in (1, 2.5, 5):
                T = m * 10 ** e
                if max(c.median, 10) < T <= c.max:
                    auto.append(T)
    return [(T, True) for T in declared] + [(T, False) for T in auto[:12] if T not in declared]


def _bunching(ctx, c):
    q, rel, cfg = qident(c.name), ctx.rel, ctx.cfg
    cands = _candidates(ctx, c)
    if not cands:
        return []
    exprs = []
    for T, _ in cands:
        exprs += [f"count_if({q} >= {0.95 * T!r} AND {q} < {T!r})", f"count_if({q} >= {T!r} AND {q} < {1.05 * T!r})",
                  f"count_if({q} >= {0.90 * T!r} AND {q} < {0.95 * T!r})"]
    row = ctx.con.execute("SELECT " + ", ".join(exprs) + f" FROM {rel}").fetchone()
    out = []
    for i, (T, declared) in enumerate(cands):
        below, above, far = [int(v or 0) for v in row[3 * i:3 * i + 3]]
        if below >= cfg.bunch_min_count and below >= cfg.bunch_ratio * max(above, 1) and below >= 1.5 * max(far, 1):
            ratio = below / (above + 1)
            out.append(Fact(type="threshold_bunching", title=f"Bunching below {fmt_num(T)} in {c.name}", columns=[c.name],
                            metrics={"column": c.name, "threshold": T, "below": below, "above": above, "below_far": far, "declared": declared},
                            severity="warning" if declared and ratio >= 3 else "notable", score=min(1.0, ratio / 10),
                            evidence_sql=(f"SELECT count_if({q} >= {0.90 * T!r} AND {q} < {0.95 * T!r}) AS band_90_95, count_if({q} >= {0.95 * T!r} AND {q} < {T!r}) AS just_below, "
                                          f"count_if({q} >= {T!r} AND {q} < {1.05 * T!r}) AS just_above FROM {rel}")))
    out.sort(key=lambda f: -f.score)
    return out[:5]
