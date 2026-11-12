"""Target leakage and label balance. Needs a label column (semantics `dataset.label`, GUI/CLI option, or a name like `target`)."""
from __future__ import annotations
from ..facts import Fact
from ..semantics import _lit
from ..util import qident


def label_kind(ctx):
    c = ctx.col(ctx.label) if ctx.label else None
    if not c:
        return None, None
    if c.kind in ("text", "boolean") or (c.kind == "numeric" and 2 <= c.distinct <= 20):
        return c, "class"
    if c.kind == "numeric" and c.distinct > 20:
        return c, "regression"
    return None, None


def _sample(ctx, *qcols):
    rows = ctx.profile.rows
    mod = max(1, rows // ctx.cfg.leak_sample_rows)
    return f"AND hash({', '.join(qcols)}) % {mod} = 0" if mod > 1 else ""


def _purity(rows):
    """rows: (bin, label, count). Returns (balanced accuracy of the best per-bin label, chance level 1/k, n).
    Balanced accuracy (mean per-class recall) stays meaningful when one class is rare."""
    bins, labels = {}, {}
    for b, y, n in rows:
        bins.setdefault(b, {})[y] = n
        labels[y] = labels.get(y, 0) + n
    n = sum(labels.values())
    if not n or len(labels) < 2:
        return 0.0, 1.0, n
    hit = {y: 0 for y in labels}
    for cnt in bins.values():
        top = max(cnt, key=cnt.get)
        hit[top] += cnt[top]
    return sum(hit[y] / labels[y] for y in labels) / len(labels), 1 / len(labels), n


def detect(ctx):
    lab, kind = label_kind(ctx)
    if not lab:
        return []
    out = _imbalance(ctx, lab) if kind == "class" else []
    cfg, rel, ql = ctx.cfg, ctx.rel, qident(lab.name)
    feats = [c for c in ctx.profile.columns if c.name != lab.name and not ctx.is_id_like(c)
             and ctx.sem.role(c.name) not in ("id", "ignore", "time", "entity") and c.kind in ("numeric", "text", "boolean")
             and not ctx.sem.excluded(c.name, "leakage") and c.distinct > 1]
    feats = sorted(feats, key=lambda c: c.nulls)[:60]
    for f in feats:
        try:
            out += _feature(ctx, lab, kind, f, ql, rel, cfg)
        except Exception as e:                     # one odd column must not stop the rest
            ctx.notes.append(f"leakage check skipped for {f.name}: {e}")
    return out


def _feature(ctx, lab, kind, f, ql, rel, cfg):
    qf, rows, out = qident(f.name), ctx.profile.rows, []
    smp = _sample(ctx, qf, ql)
    if kind == "regression":
        if f.kind != "numeric" or f.distinct <= 5:
            return []
        r = ctx.con.execute(f"SELECT corr({qf}, {ql}), count(*) FROM {rel} WHERE {qf} IS NOT NULL AND {ql} IS NOT NULL").fetchone()
        if r[0] is not None and abs(r[0]) >= 0.98:
            out.append(Fact(type="target_leakage", title=f"{f.name} nearly determines {lab.name}", columns=[f.name, lab.name],
                            metrics={"feature": f.name, "label": lab.name, "purity": abs(float(r[0])), "baseline": 0.0, "method": "absolute linear correlation", "measure": "correlation", "n": int(r[1])},
                            severity="warning" if abs(r[0]) >= 0.995 else "notable", score=abs(float(r[0])),
                            evidence_sql=f"SELECT corr({qf}, {ql}) AS pearson_r, count(*) AS n FROM {rel} WHERE {qf} IS NOT NULL AND {ql} IS NOT NULL",
                            chart={"kind": "scatter", "xlabel": f.name, "ylabel": lab.name,
                                   "sql": f"SELECT x, y FROM (SELECT {qf} AS x, {ql} AS y FROM {rel} WHERE {qf} IS NOT NULL AND {ql} IS NOT NULL) USING SAMPLE 5000 ROWS"}))
        return out
    nn = max(rows - f.nulls, 1)
    binary = lab.distinct == 2
    if f.kind == "numeric" and f.distinct > 20 and binary:
        # rank-based AUC: robust to class imbalance and needs no binning
        sql = (f"SELECT label, count(*) AS n, sum(rk) AS rank_sum FROM (SELECT {ql} AS label, rank() OVER (ORDER BY {qf}) + "
               f"(count(*) OVER (PARTITION BY {qf}) - 1) / 2.0 AS rk FROM {rel} WHERE {qf} IS NOT NULL AND {ql} IS NOT NULL {smp}) GROUP BY 1")
        rs = ctx.con.execute(sql).fetchall()
        if len(rs) == 2 and min(r[1] for r in rs) >= 30:
            (_, n0, s0), (_, n1, _s1) = rs
            auc = (float(s0) - n0 * (n0 + 1) / 2) / (n0 * n1)
            auc = max(auc, 1 - auc)
            if auc >= 0.99:
                out.append(Fact(type="target_leakage", title=f"{f.name} nearly determines {lab.name}", columns=[f.name, lab.name],
                                metrics={"feature": f.name, "label": lab.name, "purity": auc, "baseline": 0.5, "method": "rank AUC", "measure": "auc", "n": int(n0 + n1)},
                                severity="warning" if auc >= 0.999 else "notable", score=auc - 0.5, evidence_sql=sql))
        return out + _missing(ctx, lab, f, ql, qf, rel, cfg, rows, smp)
    if f.kind == "numeric" and f.distinct > 20:
        qs = [i / cfg.leak_bins for i in range(1, cfg.leak_bins)]
        edges = ctx.con.execute(f"SELECT quantile_cont({qf}, {qs!r}) FROM {rel} WHERE {qf} IS NOT NULL {smp}").fetchone()[0]
        edges = sorted(set(float(e) for e in edges if e is not None))
        if not edges:
            return out
        bin_sql = "CASE " + " ".join(f"WHEN {qf} <= {e!r} THEN {i}" for i, e in enumerate(edges)) + f" ELSE {len(edges)} END"
        method = f"{len(edges) + 1} quantile bins, balanced accuracy"
    elif f.distinct <= 1000 and nn / max(f.distinct, 1) >= 20:
        bin_sql, method = f"CAST({qf} AS VARCHAR)", "value lookup, balanced accuracy"
    else:
        bin_sql = None
    if bin_sql:
        sql = (f"SELECT {bin_sql} AS bin, {ql} AS label, count(*) AS n FROM {rel} WHERE {qf} IS NOT NULL AND {ql} IS NOT NULL {smp} GROUP BY 1, 2")
        pur, chance, n = _purity(ctx.con.execute(sql).fetchall())
        if n >= 200 and pur >= cfg.leak_purity and pur - chance >= cfg.leak_lift:
            out.append(Fact(type="target_leakage", title=f"{f.name} nearly determines {lab.name}", columns=[f.name, lab.name],
                            metrics={"feature": f.name, "label": lab.name, "purity": pur, "baseline": chance, "method": method, "measure": "balanced_accuracy", "n": n},
                            severity="warning" if pur >= 0.995 else "notable", score=pur - chance, evidence_sql=sql + " ORDER BY 1, 3 DESC"))
    return out + _missing(ctx, lab, f, ql, qf, rel, cfg, rows, smp)


def _missing(ctx, lab, f, ql, qf, rel, cfg, rows, smp):
    out = []
    if 0.01 <= f.nulls / rows <= 0.99:
        sql = f"SELECT ({qf} IS NULL) AS bin, {ql} AS label, count(*) AS n FROM {rel} WHERE {ql} IS NOT NULL {smp} GROUP BY 1, 2"
        pur, base, n = _purity(ctx.con.execute(sql).fetchall())
        if n >= 200 and pur >= 0.95 and pur - base >= cfg.leak_lift:
            out.append(Fact(type="leakage_missingness", title=f"Missingness of {f.name} predicts {lab.name}", columns=[f.name, lab.name],
                            metrics={"feature": f.name, "label": lab.name, "purity": pur, "baseline": base, "null_share": f.nulls / rows, "n": n},
                            severity="warning" if pur >= 0.99 else "notable", score=pur - base, evidence_sql=sql + " ORDER BY 1, 3 DESC"))
    return out


def _imbalance(ctx, lab):
    ql, rel, cfg = qident(lab.name), ctx.rel, ctx.cfg
    sql = f"SELECT CAST({ql} AS VARCHAR) AS x, count(*) AS y FROM {rel} WHERE {ql} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"
    rows = ctx.con.execute(sql).fetchall()
    if len(rows) < 2:
        return []
    n = sum(r[1] for r in rows)
    minority = rows[-1]
    share = minority[1] / n
    if share >= cfg.imbalance_notable:
        return []
    return [Fact(type="class_imbalance", title=f"Class imbalance in {lab.name}", columns=[lab.name],
                 metrics={"label": lab.name, "classes": len(rows), "minority": minority[0], "minority_share": share,
                          "ratio": rows[0][1] / minority[1], "top": [[r[0], r[1]] for r in rows[:5]]},
                 severity="warning" if share < cfg.imbalance_warning else "notable", score=1 - share, evidence_sql=sql,
                 chart={"kind": "bar", "xlabel": lab.name, "ylabel": "rows", "sql": sql})]
