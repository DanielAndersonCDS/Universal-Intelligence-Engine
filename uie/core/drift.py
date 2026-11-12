"""Dataset drift: PSI, two-sample KS, chi-square (with Cramer's V), null-rate and schema changes between a reference and a current dataset."""
from __future__ import annotations
import numpy as np
from . import provenance, stats
from .config import Config
from .facts import Fact
from .ingest import open_pair
from .profile import compute_profile
from .semantics import Semantics, build_clean_view
from .subreport import SubReport
from .util import qident


def _sample(con, view, cols, limit):
    n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
    mod = max(1, n // limit)
    names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()][:8]
    where = f"WHERE hash({', '.join(qident(c) for c in names)}) % {mod} = 0" if mod > 1 else ""
    rows = con.execute(f"SELECT {', '.join(qident(c) for c in cols)} FROM {view} {where}").fetchall()
    return np.array(rows, dtype=float).reshape(len(rows), len(cols))


def _flag(psi, ks, null_change, cfg):
    if psi is not None and psi >= cfg.psi_warning:
        return "warning"
    if (psi is not None and psi >= cfg.psi_notable) or (ks is not None and ks >= cfg.ks_notable) or abs(null_change) >= 0.10:
        return "notable"
    return None


def drift(left, right, label=None, cfg: Config | None = None, sem: Semantics | None = None, table_left=None, table_right=None,
          ignore=None, progress=lambda s, p=0: None, command=None, con_views=None) -> SubReport:
    """Compare `right` (current) against `left` (reference). `con_views`=(con, a, b) reuses an already-open pair."""
    cfg = cfg or Config()
    progress("Opening datasets", 5)
    con, a, b = con_views or open_pair(left, right, table_left, table_right)
    from .util import qident as q
    build_clean_view(con, "left_raw", "left_data", sem, q)
    build_clean_view(con, "right_raw", "right_data", sem, q)
    progress("Profiling", 20)
    pa, pb = compute_profile(con.cursor(), "left_data"), compute_profile(con.cursor(), "right_data")
    ca, cb = {c.name: c for c in pa.columns}, {c.name: c for c in pb.columns}
    ignore = set(ignore or [])
    facts, rows, schema_msgs = [], [], []
    for n in ca:
        if n not in cb:
            facts.append(Fact(type="drift_schema", title=f"Column removed: {n}", columns=[n], severity="warning", score=0.9,
                              metrics={"column": n, "change": "removed", "message": f"Column `{n}` exists in the reference but not in the current dataset."}))
    for n in cb:
        if n not in ca:
            facts.append(Fact(type="drift_schema", title=f"Column added: {n}", columns=[n], severity="notable", score=0.5,
                              metrics={"column": n, "change": "added", "message": f"Column `{n}` is new in the current dataset."}))
    common = [n for n in ca if n in cb and n not in ignore]
    for n in common:
        if ca[n].kind != cb[n].kind:
            facts.append(Fact(type="drift_schema", title=f"Type changed: {n}", columns=[n], severity="warning", score=0.9,
                              metrics={"column": n, "change": "type", "message": f"`{n}` changed type from {ca[n].dtype} to {cb[n].dtype}."}))
    num = [n for n in common if ca[n].kind == "numeric" and cb[n].kind == "numeric" and (ca[n].distinct > 1 or cb[n].distinct > 1)]
    if num:
        progress("Numeric drift", 45)
        A, B = _sample(con, "left_data", num, cfg.drift_sample_rows), _sample(con, "right_data", num, cfg.drift_sample_rows)
        for j, n in enumerate(num):
            x, y = A[:, j], B[:, j]
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x) < 30 or len(y) < 30:
                continue
            edges = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))
            cx = np.bincount(np.searchsorted(edges, x), minlength=len(edges) + 1)
            cy = np.bincount(np.searchsorted(edges, y), minlength=len(edges) + 1)
            psi = stats.psi(cx, cy)
            d, p = stats.ks_2samp(x, y)
            sd = ca[n].std or 0.0
            shift = (cb[n].mean - ca[n].mean) / sd if sd and cb[n].mean is not None and ca[n].mean is not None else 0.0
            nc = cb[n].nulls / max(pb.rows, 1) - ca[n].nulls / max(pa.rows, 1)
            rows.append({"column": n, "kind": "numeric", "psi": psi, "ks": d, "ks_p": p, "chi2_p": None, "cramers_v": None, "mean_a": ca[n].mean,
                         "mean_b": cb[n].mean, "mean_shift": shift, "null_a": ca[n].nulls / max(pa.rows, 1), "null_b": cb[n].nulls / max(pb.rows, 1),
                         "null_change": nc})
    cat = [n for n in common if ca[n].kind in ("text", "boolean") and cb[n].kind in ("text", "boolean") and max(ca[n].distinct, cb[n].distinct) <= 5000
           and not (pa.rows >= 500 and ca[n].distinct >= 0.97 * pa.rows)]
    progress("Categorical drift", 70)
    for n in cat:
        qn = qident(n)
        rs = con.execute(f"SELECT v, sum(l), sum(r) FROM (SELECT CAST({qn} AS VARCHAR) v, count(*) l, 0 r FROM left_data WHERE {qn} IS NOT NULL GROUP BY 1 "
                         f"UNION ALL SELECT CAST({qn} AS VARCHAR), 0, count(*) FROM right_data WHERE {qn} IS NOT NULL GROUP BY 1) GROUP BY 1 ORDER BY 2 + 3 DESC").fetchall()
        if len(rs) < 2:
            continue
        top, rest = rs[:cfg.drift_max_categories], rs[cfg.drift_max_categories:]
        cl = [r[1] for r in top] + ([sum(r[1] for r in rest)] if rest else [])
        cr = [r[2] for r in top] + ([sum(r[2] for r in rest)] if rest else [])
        chi2, df, p, v = stats.chi2_two_sample(cl, cr)
        psi = stats.psi(cl, cr)
        nc = cb[n].nulls / max(pb.rows, 1) - ca[n].nulls / max(pa.rows, 1)
        rows.append({"column": n, "kind": "categorical", "psi": psi, "ks": None, "ks_p": None, "chi2_p": p, "cramers_v": v, "mean_a": None, "mean_b": None,
                     "mean_shift": None, "null_a": ca[n].nulls / max(pa.rows, 1), "null_b": cb[n].nulls / max(pb.rows, 1), "null_change": nc})
    for r in rows:
        sev = _flag(r["psi"], r["ks"], r["null_change"], cfg)
        r["flag"] = (sev or "").upper()
        if not sev:
            continue
        n, qn = r["column"], qident(r["column"])
        ev = (f"SELECT 'reference' AS dataset, count(*) AS rows, avg(TRY_CAST({qn} AS DOUBLE)) AS mean, count(*) FILTER (WHERE {qn} IS NULL) AS nulls FROM left_data UNION ALL "
              f"SELECT 'current', count(*), avg(TRY_CAST({qn} AS DOUBLE)), count(*) FILTER (WHERE {qn} IS NULL) FROM right_data")
        m = {"column": n, "kind": r["kind"], "psi": r["psi"], "ks": r["ks"] or 0.0, "mean_shift": r["mean_shift"] or 0.0, "p": r["chi2_p"] if r["chi2_p"] is not None else 1.0,
             "cramers_v": r["cramers_v"] or 0.0, "null_change": r["null_change"]}
        facts.append(Fact(type="drift", title=f"Drift in {n}", columns=[n], metrics=m, severity=sev, score=min(1.0, r["psi"]), evidence_sql=ev))
    if label and label in ca and label in cb:
        pass   # the label is just another column; it appears in the table with its own PSI
    rows.sort(key=lambda r: -(r["psi"] or 0))
    facts.sort(key=lambda f: (0 if f.severity == "warning" else 1 if f.severity == "notable" else 2, -f.score))
    for i, f in enumerate(facts, 1):
        f.id = f"D{i:03d}"
    cols = ["column", "kind", "flag", "psi", "ks_d", "ks_p", "chi2_p", "cramers_v", "mean_ref", "mean_cur", "mean_shift_std", "null_ref", "null_cur"]
    table = [[r["column"], r["kind"], r["flag"], r["psi"], r["ks"], r["ks_p"], r["chi2_p"], r["cramers_v"], r["mean_a"], r["mean_b"], r["mean_shift"], r["null_a"], r["null_b"]] for r in rows]
    summary = {"reference": a.name, "current": b.name, "rows_reference": pa.rows, "rows_current": pb.rows, "columns_compared": len(rows),
               "flagged_warning": sum(1 for f in facts if f.severity == "warning"), "flagged_notable": sum(1 for f in facts if f.severity == "notable"),
               "psi_bands": "<0.10 stable, 0.10-0.25 moderate, >=0.25 significant",
               "sampling": f"numeric PSI/KS use up to {cfg.drift_sample_rows:,} rows per side (deterministic hash sample)"}
    r = SubReport(kind="drift", title=f"Drift: {a.name} (reference) vs {b.name} (current)", summary=summary, facts=facts, tables={"Column drift": (cols, table)},
                  setup_sql=[a.source_sql(), b.source_sql()])
    r.provenance = provenance.build([provenance.input_record(a, "reference", cfg.hash_data, progress), provenance.input_record(b, "current", cfg.hash_data, progress)],
                                    {"label": label, "ignore": sorted(ignore)}, cfg, sem if sem and (sem.columns or sem.dataset) else None, None, facts, command)
    progress("Done", 100)
    return r
