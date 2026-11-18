"""Do a numeric measure's group means differ across the levels of a categorical column?
Welch t-test (2 groups) or one-way ANOVA (3+), confirmed with Kruskal-Wallis on a deterministic sample, Benjamini-Hochberg
corrected across every test run, and reported only when the effect size is at least medium."""
import numpy as np
from ..facts import Fact
from ..util import qident
from .. import stats

MIN_N = 30


def detect(ctx):
    cfg, rel = ctx.cfg, ctx.rel
    ms = sorted([c for c in ctx.measures("grouptests") if c.distinct > 5], key=lambda c: c.nulls)[:12]
    dims = sorted([c for c in ctx.profile.columns if c.kind == "text" and 2 <= c.distinct <= 20 and not ctx.is_id_like(c)
                   and c.name != ctx.group_by and ctx.sem.role(c.name) not in ("ignore", "id") and c.nulls < 0.5 * ctx.profile.rows],
                  key=lambda c: c.nulls)[:4]
    if not ms or not dims:
        return []
    tests = []
    for d in dims:
        qd = qident(d.name)
        sel = ", ".join(f"count({qident(m.name)}), avg({qident(m.name)}), var_samp({qident(m.name)})" for m in ms)
        rows = ctx.con.execute(f"SELECT CAST({qd} AS VARCHAR), {sel} FROM {rel} WHERE {qd} IS NOT NULL GROUP BY 1").fetchall()
        sample = ctx.sample_rows([d.name] + [m.name for m in ms], 60000)
        gl = np.array([("" if r[0] is None else str(r[0])) for r in sample])
        for j, m in enumerate(ms):
            g = [(r[0], r[1 + 3 * j], r[2 + 3 * j], r[3 + 3 * j]) for r in rows if r[1 + 3 * j] and r[1 + 3 * j] >= MIN_N and r[3 + 3 * j] is not None]
            if len(g) < 2:
                continue
            ns, means, vs = [x[1] for x in g], [float(x[2]) for x in g], [float(x[3]) for x in g]
            if len(g) == 2:
                t, df, p, eff = stats.welch_t(ns[0], means[0], vs[0], ns[1], means[1], vs[1])
                test, stat, effect, ename, big = "Welch t-test", ("t", abs(t)), abs(eff), "Cohen's d", abs(eff) >= 0.5
            else:
                f, d1, d2, p, eta = stats.anova_oneway(ns, means, vs)
                test, stat, effect, ename, big = "one-way ANOVA", ("F", f), eta, "eta-squared", eta >= 0.06
            vals = np.array([r[1 + j] for r in sample], dtype=float)
            labels = {x[0] for x in g}
            groups = [vals[(gl == lab) & np.isfinite(vals)] for lab in labels]
            kw_p = stats.kruskal([x for x in groups if len(x) >= 5])[2] if sum(len(x) >= 5 for x in groups) >= 2 else 1.0
            tests.append({"m": m, "d": d, "g": g, "test": test, "stat": stat, "p": p, "effect": effect, "ename": ename, "big": big, "kw": kw_p})
    if not tests:
        return []
    adj = stats.bh_adjust([t["p"] for t in tests])
    out = []
    for t, pa in zip(tests, adj):
        if not (t["big"] and pa < 0.001 and t["kw"] < 0.01):
            continue
        hi, lo = max(t["g"], key=lambda x: x[2]), min(t["g"], key=lambda x: x[2])
        qm, qd = qident(t["m"].name), qident(t["d"].name)
        out.append(Fact(
            type="group_test", title=f"{t['m'].name} differs across {t['d'].name}", columns=[t["m"].name, t["d"].name],
            metrics={"measure": t["m"].name, "dimension": t["d"].name, "test": t["test"], "stat_name": t["stat"][0], "stat": float(t["stat"][1]),
                     "p": float(t["p"]), "p_adj": float(pa), "kruskal_p": float(t["kw"]), "effect": float(t["effect"]), "effect_name": t["ename"],
                     "groups": len(t["g"]), "hi_group": hi[0], "hi_mean": float(hi[2]), "lo_group": lo[0], "lo_mean": float(lo[2]), "tests_run": len(tests)},
            severity="notable" if t["effect"] >= (0.8 if t["ename"] == "Cohen's d" else 0.14) else "info", score=min(1.0, t["effect"] * (1 if t["ename"] == "Cohen's d" else 3)),
            evidence_sql=(f"SELECT {qd} AS grp, count({qm}) AS n, avg({qm}) AS mean, stddev_samp({qm}) AS std FROM {rel} WHERE {qd} IS NOT NULL GROUP BY 1 ORDER BY 3 DESC"),
            chart={"kind": "bar", "xlabel": t["d"].name, "ylabel": f"mean of {t['m'].name}",
                   "sql": f"SELECT CAST({qd} AS VARCHAR) AS x, avg({qm}) AS y FROM {rel} WHERE {qd} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"}))
    out.sort(key=lambda f: -f.score)
    return out[:8]
