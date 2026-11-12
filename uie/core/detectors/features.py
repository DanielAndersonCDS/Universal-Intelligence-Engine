"""Feature health for ML datasets: near-constant columns and high-cardinality categoricals."""
from ..facts import Fact
from ..util import qident


def detect(ctx):
    cfg, rel, rows = ctx.cfg, ctx.rel, ctx.profile.rows
    skip = {ctx.label}
    out = []
    usable = [c for c in ctx.profile.columns if c.name not in skip and not ctx.is_id_like(c)
              and ctx.sem.role(c.name) not in ("id", "ignore", "time", "entity")]
    nums = [c for c in usable if c.kind == "numeric" and c.distinct > 1 and c.median is not None][:60]
    if nums:
        row = ctx.con.execute("SELECT " + ", ".join(
            f"count_if(abs({qident(c.name)} - ({c.median!r})) <= 1e-12 * greatest(1, abs({c.median!r}))), count({qident(c.name)})" for c in nums) + f" FROM {rel}").fetchone()
        for i, c in enumerate(nums):
            k, nn = row[2 * i], row[2 * i + 1]
            if nn and k / nn >= cfg.near_constant:
                q = qident(c.name)
                out.append(Fact(type="near_constant", title=f"Near-constant column {c.name}", columns=[c.name],
                                metrics={"column": c.name, "value": c.median, "share": k / nn},
                                severity="notable" if k / nn >= 0.999 else "info", score=k / nn * 0.5,
                                evidence_sql=f"SELECT {q}, count(*) AS rows FROM {rel} GROUP BY 1 ORDER BY 2 DESC LIMIT 10"))
    for c in [c for c in usable if c.kind == "text" and c.distinct > 1][:30]:
        nn = rows - c.nulls
        if c.top and nn and c.top[0][1] / nn >= cfg.near_constant:
            q = qident(c.name)
            out.append(Fact(type="near_constant", title=f"Near-constant column {c.name}", columns=[c.name],
                            metrics={"column": c.name, "value": c.top[0][0], "share": c.top[0][1] / nn},
                            severity="info", score=0.3, evidence_sql=f"SELECT {q}, count(*) AS rows FROM {rel} GROUP BY 1 ORDER BY 2 DESC LIMIT 10"))
        elif c.distinct >= cfg.high_card:
            q = qident(c.name)
            sql = (f"SELECT coalesce(sum(n) FILTER (WHERE n < {cfg.rare_count}), 0) / sum(n) AS rare_share, count(*) AS distinct_values, max(n) / sum(n) AS top_share "
                   f"FROM (SELECT count(*) AS n FROM {rel} WHERE {q} IS NOT NULL GROUP BY {q})")
            rare, dist, top = ctx.con.execute(sql).fetchone()
            out.append(Fact(type="high_cardinality", title=f"High-cardinality column {c.name}", columns=[c.name],
                            metrics={"column": c.name, "distinct": int(dist), "rows": rows, "rare_share": float(rare or 0), "top_share": float(top or 0),
                                     "rare_count": cfg.rare_count},
                            severity="notable" if (rare or 0) >= 0.2 else "info", score=min(0.5, float(rare or 0)), evidence_sql=sql))
    return out
