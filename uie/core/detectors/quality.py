from ..facts import Fact
from ..util import qident
from ..semantics import clean_condition


def detect(ctx):
    p, rel, rows, out = ctx.profile, ctx.rel, ctx.profile.rows, []
    if rows == 0:
        return [Fact(type="quality_empty", title="Dataset is empty", metrics={"rows": 0}, severity="warning", score=1)]
    for name, cnt in ctx.sentinel_counts.items():
        if cnt:
            q = qident(name)
            cp = ctx.col(name)
            cond = clean_condition(ctx.sem.columns[name], q, cp.kind == "numeric", cp.kind == "text")
            out.append(Fact(type="quality_sentinel", title=f"Sentinel values in {name}",
                            metrics={"column": name, "count": cnt, "rule": ctx.sem.columns[name].rule_text()},
                            severity="info", score=0.05, columns=[name],
                            evidence_sql=f"SELECT count(*) AS cells_treated_as_missing FROM raw WHERE {cond}"))
    for c in p.columns:
        q = qident(c.name)
        real_nulls = max(0, c.nulls - ctx.sentinel_counts.get(c.name, 0))   # sentinel cells are intentional, not missing data
        frac = real_nulls / rows
        if ctx.sem.role(c.name) == "ignore":
            continue
        if frac >= 0.05:
            out.append(Fact(type="quality_nulls", title=f"Missing values in {c.name}",
                            metrics={"column": c.name, "nulls": real_nulls, "rows": rows, "frac": frac},
                            severity="warning" if frac >= 0.30 else "notable", score=frac, columns=[c.name],
                            evidence_sql=f"SELECT count(*) FILTER (WHERE {q} IS NULL) AS missing, count(*) AS total FROM {rel}"))
        if c.distinct <= 1 and c.nulls < rows and c.kind != "other":
            val = ctx.con.execute(f"SELECT any_value({q}) FROM {rel} WHERE {q} IS NOT NULL").fetchone()[0]
            out.append(Fact(type="quality_constant", title=f"Constant column {c.name}",
                            metrics={"column": c.name, "value": str(val)}, severity="info", score=0.3,
                            columns=[c.name], evidence_sql=f"SELECT DISTINCT {q} FROM {rel}"))
        elif ctx.is_id_like(c):
            out.append(Fact(type="quality_id_like", title=f"Identifier-like column {c.name}",
                            metrics={"column": c.name}, severity="info", score=0.1, columns=[c.name],
                            evidence_sql=f"SELECT count(*) AS rows, count(DISTINCT {q}) AS distinct_values FROM {rel}"))
    try:
        d = ctx.con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {rel})").fetchone()[0]
        dupes = rows - d
        if dupes > 0:
            fr = dupes / rows
            out.append(Fact(type="quality_duplicates", title="Duplicate rows",
                            metrics={"dupes": dupes, "frac": fr, "rows": rows},
                            severity="warning" if fr >= 0.05 else ("notable" if fr >= 0.0005 else "info"), score=min(1, fr * 10),
                            evidence_sql=f"SELECT *, count(*) AS copies FROM {rel} GROUP BY ALL HAVING count(*) > 1 ORDER BY copies DESC LIMIT 100"))
    except Exception:
        pass
    if not ctx.all_text:
        for c in [c for c in p.columns if c.kind == "text"][:20]:
            q = qident(c.name)
            ok, nn = ctx.con.execute(f"SELECT count(try_cast({q} AS DOUBLE)), count({q}) FROM {rel}").fetchone()
            if nn and ok / nn >= 0.95:
                out.append(Fact(type="quality_text_numeric", title=f"Numbers stored as text in {c.name}",
                                metrics={"column": c.name, "frac": ok / nn}, severity="notable", score=0.5, columns=[c.name],
                                evidence_sql=f"SELECT {q} FROM {rel} WHERE try_cast({q} AS DOUBLE) IS NULL AND {q} IS NOT NULL LIMIT 100"))
    return out
