"""Cross-group contrast: how much of each measure's variance is explained by the chosen group column."""
from ..facts import Fact
from ..semantics import _lit
from ..util import qident


def contrast(ctx, gcol, values):
    cols = [c for c in ctx.measures("trends") if c.name != gcol][:12]
    if len(values) < 2 or not cols:
        return []
    qg, rel = qident(gcol), ctx.rel
    sel = ", ".join(f"count({qident(c.name)}), avg({qident(c.name)}), var_pop({qident(c.name)})" for c in cols)
    rows = ctx.con.execute(f"SELECT {qg}, {sel} FROM {rel} WHERE {qg} IN ({', '.join(_lit(v) for v in values)}) GROUP BY 1").fetchall()
    out = []
    for j, c in enumerate(cols):
        stats = [(str(r[0]), r[1 + 3 * j], r[2 + 3 * j], r[3 + 3 * j]) for r in rows if r[1 + 3 * j] and r[2 + 3 * j] is not None]
        if len(stats) < 2:
            continue
        N = sum(n for _, n, _, _ in stats)
        grand = sum(n * m for _, n, m, _ in stats) / N
        ssb = sum(n * (m - grand) ** 2 for _, n, m, _ in stats)
        ssw = sum(n * (v or 0.0) for _, n, _, v in stats)
        if ssb + ssw <= 0:
            continue
        eta2 = ssb / (ssb + ssw)
        if eta2 < ctx.cfg.group_eta2:
            continue
        hi, lo = max(stats, key=lambda s: s[2]), min(stats, key=lambda s: s[2])
        q = qident(c.name)
        out.append(Fact(type="group_contrast", title=f"{c.name} differs across {gcol}", columns=[c.name, gcol],
                        metrics={"column": c.name, "group_column": gcol, "eta2": eta2, "hi_group": hi[0], "hi_mean": hi[2],
                                 "lo_group": lo[0], "lo_mean": lo[2], "groups": len(stats)},
                        severity="notable" if eta2 >= 0.26 else "info", score=eta2,
                        evidence_sql=f"SELECT {qg} AS grp, count({q}) AS n, avg({q}) AS mean, stddev_samp({q}) AS std FROM {rel} GROUP BY 1 ORDER BY 3 DESC",
                        chart={"kind": "bar", "xlabel": gcol, "ylabel": f"mean of {c.name}",
                               "sql": f"SELECT {qg} AS x, avg({q}) AS y FROM {rel} WHERE {qg} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"}))
    return out
