from ..facts import Fact
from ..util import qident


def detect(ctx):
    p, rel, rows = ctx.profile, ctx.rel, ctx.profile.rows
    cats = sorted([c for c in p.columns if c.kind == "text" and 3 <= c.distinct <= 50
                   and not ctx.is_id_like(c) and ctx.sem.role(c.name) not in ("ignore", "id") and c.nulls < 0.5 * rows], key=lambda c: c.nulls)[:6]
    nums = [c for c in ctx.measures("segments") if c.min is not None and c.min >= 0 and c.distinct > 10][:6]
    if not cats or not nums:
        return []
    out = []
    for g in cats:
        qg = qident(g.name)
        gx = f"coalesce(CAST({qg} AS VARCHAR), '(missing)')"
        sums = ", ".join(f"sum({qident(n.name)})" for n in nums)
        rs = ctx.con.execute(f"SELECT {gx} AS g, count(*) AS n, {sums} FROM {rel} GROUP BY 1").fetchall()
        total_n = sum(r[1] for r in rs)
        for j, n in enumerate(nums):
            tot = sum(float(r[2 + j] or 0) for r in rs)
            if tot <= 0 or len(rs) < 3:
                continue
            top = max(rs, key=lambda r: float(r[2 + j] or 0))
            vs, rsh = float(top[2 + j] or 0) / tot, top[1] / total_n
            if vs >= 0.35 and vs / rsh >= 1.5:
                qn = qident(n.name)
                out.append(Fact(
                    type="concentration", title=f"{n.name} concentrated in {g.name}", columns=[n.name, g.name],
                    metrics={"column": n.name, "group_column": g.name, "group_value": str(top[0]),
                             "value_share": vs, "row_share": rsh, "groups": len(rs)},
                    severity="notable", score=vs * (1 - rsh),
                    evidence_sql=f"SELECT {gx} AS {qg}, count(*) AS rows, sum({qn}) AS total FROM {rel} GROUP BY 1 ORDER BY 3 DESC",
                    chart={"kind": "bar", "xlabel": g.name, "ylabel": f"sum of {n.name}",
                           "sql": f"SELECT {gx} AS x, sum({qn}) AS y FROM {rel} GROUP BY 1 ORDER BY 2 DESC LIMIT 12"}))
    return out
