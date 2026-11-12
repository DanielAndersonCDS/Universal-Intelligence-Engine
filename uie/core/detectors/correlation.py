from ..facts import Fact
from ..util import qident


def detect(ctx):
    cols = sorted([c for c in ctx.measures("correlation") if c.distinct > 5], key=lambda c: c.nulls)[:ctx.cfg.corr_max_cols]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:] if not ctx.is_derived_pair(a.name, b.name)]
    if not pairs:
        return []
    rel, thr = ctx.rel, ctx.cfg.corr_threshold
    row = ctx.con.execute("SELECT " + ", ".join(
        f"corr({qident(a.name)}, {qident(b.name)}), count_if({qident(a.name)} IS NOT NULL AND {qident(b.name)} IS NOT NULL)"
        for a, b in pairs) + f" FROM {rel}").fetchone()
    found = []
    for i, (a, b) in enumerate(pairs):
        r, n = row[2 * i], row[2 * i + 1]
        if r is not None and abs(r) >= thr:
            found.append((abs(r), float(r), int(n), a, b))
    found.sort(key=lambda t: -t[0])
    out = []
    for ar, r, n, a, b in found[:12]:
        qa, qb = qident(a.name), qident(b.name)
        out.append(Fact(
            type="near_duplicate" if ar >= 0.999 else "correlation",
            title=f"{a.name} vs {b.name}", columns=[a.name, b.name],
            metrics={"a": a.name, "b": b.name, "r": r, "n": n, "method": "pearson"},
            severity="notable" if ar >= 0.85 else "info", score=ar,
            evidence_sql=f"SELECT corr({qa}, {qb}) AS pearson_r, count(*) AS n FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL",
            chart={"kind": "scatter", "xlabel": a.name, "ylabel": b.name,
                   "sql": f"SELECT x, y FROM (SELECT {qa} AS x, {qb} AS y FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL) USING SAMPLE 5000 ROWS"}))
    return out
