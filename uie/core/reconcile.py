"""Row-level reconciliation of two datasets by key: missing, extra and mismatched amounts, with a bridge that
explains the difference in totals."""
from __future__ import annotations
from . import provenance
from .config import Config
from .facts import Fact
from .ingest import open_pair
from .subreport import SubReport
from .util import qident
import re

NUMERIC = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "REAL"}


def _is_num(t):
    t = str(t).upper()
    return t in NUMERIC or t.startswith("DECIMAL")


def _knorm(col):
    """Normalised key: text, trimmed, upper-cased, trailing '.0' removed (so 12 and 12.0 and ' 12' match)."""
    return f"coalesce(upper(trim(regexp_replace(CAST({qident(col)} AS VARCHAR), '\\.0+$', ''))), '<NULL>')"


def reconcile(left, right, key, compare=None, tol=None, rel_tol=None, cfg: Config | None = None, table_left=None, table_right=None,
              progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    tol = cfg.recon_tolerance if tol is None else tol
    rel_tol = cfg.recon_rel_tolerance if rel_tol is None else rel_tol
    key = [key] if isinstance(key, str) else list(key)
    progress("Opening datasets", 5)
    con, a, b = open_pair(left, right, table_left, table_right)
    con.execute("CREATE VIEW left_data AS SELECT * FROM left_raw")
    con.execute("CREATE VIEW right_data AS SELECT * FROM right_raw")
    dl = {r[0]: str(r[1]) for r in con.execute("DESCRIBE left_data").fetchall()}
    dr = {r[0]: str(r[1]) for r in con.execute("DESCRIBE right_data").fetchall()}
    for k in key:
        if k not in dl or k not in dr:
            raise ValueError(f"Key column '{k}' must exist in both datasets (left: {k in dl}, right: {k in dr}).")
    if compare is None:
        compare = [c for c in dl if c in dr and c not in key and _is_num(dl[c]) and _is_num(dr[c])][:20]
    compare = [compare] if isinstance(compare, str) else list(compare)
    for c in compare:
        if c not in dl or c not in dr:
            raise ValueError(f"Compare column '{c}' must exist in both datasets.")
    ln, rn = a.name, b.name
    kx = [_knorm(k) for k in key]
    num = lambda c: f"TRY_CAST({qident(c)} AS DOUBLE)"

    def side(view):
        sel = ", ".join(f"{e} AS k{i}" for i, e in enumerate(kx))
        agg = "".join(f", sum({num(c)}) AS c{i}" for i, c in enumerate(compare))
        grp = ", ".join(str(i + 1) for i in range(len(key)))
        return f"SELECT {sel}, count(*) AS n{agg} FROM {view} GROUP BY {grp}"

    on = " AND ".join(f"L.k{i} = R.k{i}" for i in range(len(key)))
    kcols = ", ".join(f"coalesce(L.k{i}, R.k{i}) AS k{i}" for i in range(len(key)))
    vals = "".join(f", L.c{i} AS l{i}, R.c{i} AS r{i}" for i in range(len(compare)))

    def diff_cond(i):
        return (f"abs(coalesce(l{i}, 0) - coalesce(r{i}, 0)) > greatest({tol!r}, {rel_tol!r} * greatest(abs(coalesce(l{i}, 0)), abs(coalesce(r{i}, 0))))")

    any_diff = " OR ".join(diff_cond(i) for i in range(len(compare))) or "FALSE"
    ctes = (f"WITH L AS ({side('left_data')}), R AS ({side('right_data')}), J AS (SELECT {kcols}, L.n AS ln, R.n AS rn{vals} FROM L FULL OUTER JOIN R ON {on}), "
            f"S AS (SELECT *, CASE WHEN ln IS NULL THEN 'EXTRA_IN_RIGHT' WHEN rn IS NULL THEN 'MISSING_IN_RIGHT' WHEN {any_diff} THEN 'MISMATCH' ELSE 'MATCH' END AS status FROM J)")
    progress("Matching keys", 40)
    cnt = dict(con.execute(f"{ctes} SELECT status, count(*) FROM S GROUP BY 1").fetchall())
    miss, extra, mism, match = (int(cnt.get(k, 0)) for k in ("MISSING_IN_RIGHT", "EXTRA_IN_RIGHT", "MISMATCH", "MATCH"))
    rows_l = con.execute("SELECT count(*) FROM left_data").fetchone()[0]
    rows_r = con.execute("SELECT count(*) FROM right_data").fetchone()[0]
    dups = {}
    for nm, view in (("left", "left_data"), ("right", "right_data")):
        d = con.execute(f"WITH S AS ({side(view)}) SELECT count(*) FILTER (WHERE n > 1), coalesce(sum(n) FILTER (WHERE n > 1), 0) FROM S").fetchone()
        dups[nm] = (int(d[0] or 0), int(d[1] or 0))
    sums = {}
    if compare:
        agg = ", ".join(f"coalesce(sum(l{i}) FILTER (WHERE status = 'MISSING_IN_RIGHT'), 0), coalesce(sum(r{i}) FILTER (WHERE status = 'EXTRA_IN_RIGHT'), 0), "
                        f"coalesce(sum(coalesce(l{i}, 0) - coalesce(r{i}, 0)) FILTER (WHERE status = 'MISMATCH'), 0), "
                        f"coalesce(sum(coalesce(l{i}, 0) - coalesce(r{i}, 0)) FILTER (WHERE status = 'MATCH'), 0), "
                        f"coalesce(sum(l{i}), 0), coalesce(sum(r{i}), 0)" for i in range(len(compare)))
        row = con.execute(f"{ctes} SELECT {agg} FROM S").fetchone()
        for i, c in enumerate(compare):
            ol, orr, mm, res, tl, tr = (float(v) for v in row[6 * i:6 * i + 6])
            sums[c] = {"only_left": ol, "only_right": orr, "mismatch": mm, "residue": res, "total_left": tl, "total_right": tr}
    progress("Collecting exceptions", 70)
    sel_cols = ", ".join(f"k{i}" for i in range(len(key))) + ", ln, rn" + "".join(f", l{i}, r{i}, coalesce(l{i}, 0) - coalesce(r{i}, 0) AS d{i}" for i in range(len(compare)))
    order = "abs(d0) DESC NULLS LAST" if compare else "1"
    exc_sql = f"{ctes} SELECT status, {sel_cols} FROM S WHERE status <> 'MATCH' ORDER BY status, {order}"
    exc_cols = ["status"] + key + [f"rows ({ln})", f"rows ({rn})"] + [x for c in compare for x in (f"{c} ({ln})", f"{c} ({rn})", f"{c} diff")]
    exc_rows = con.execute(exc_sql + f" LIMIT {cfg.recon_max_exceptions}").fetchall()
    exc_rows = [list(r) for r in exc_rows]
    facts = []

    def ev(where):
        return f"{ctes} SELECT status, {sel_cols} FROM S WHERE {where} ORDER BY {order} LIMIT 1000"

    def amt(k):
        return {c: sums[c][k] for c in compare} if compare else {}

    base = {"left": ln, "right": rn}
    if miss:
        facts.append(Fact(type="recon_missing", title="Keys missing in right", columns=key, metrics={**base, "count": miss, "amounts": amt("only_left")},
                          severity="warning", score=min(1, miss / max(len(exc_rows), 1) + 0.5), evidence_sql=ev("status = 'MISSING_IN_RIGHT'")))
    if extra:
        facts.append(Fact(type="recon_extra", title="Keys extra in right", columns=key, metrics={**base, "count": extra, "amounts": amt("only_right")},
                          severity="warning", score=min(1, extra / max(len(exc_rows), 1) + 0.5), evidence_sql=ev("status = 'EXTRA_IN_RIGHT'")))
    if mism:
        facts.append(Fact(type="recon_mismatch", title="Mismatched amounts", columns=key + compare,
                          metrics={**base, "count": mism, "amounts": amt("mismatch")}, severity="warning", score=0.8, evidence_sql=ev("status = 'MISMATCH'")))
    for nm, (k, r) in dups.items():
        if k:
            facts.append(Fact(type="recon_dupkeys", title=f"Duplicate keys in {nm}", columns=key, metrics={"side": ln if nm == "left" else rn, "keys": k, "rows": r},
                              severity="notable", score=0.4,
                              evidence_sql=f"SELECT {', '.join(kx)}, count(*) AS rows FROM {nm}_data GROUP BY ALL HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 1000"))
    bridge_rows = []
    for c in compare:
        s = sums[c]
        diff = s["total_left"] - s["total_right"]
        check = diff - (s["only_left"] - s["only_right"] + s["mismatch"] + s["residue"])
        bridge_rows.append([c, s["total_left"], s["total_right"], diff, s["only_left"], s["only_right"], s["mismatch"], s["residue"], check])
        if abs(diff) > tol:
            facts.append(Fact(type="recon_bridge", title=f"Total difference in {c}", columns=[c],
                              metrics={**base, "column": c, **s, "diff": diff}, severity="notable" if abs(diff) > 10 * tol else "info", score=0.5,
                              evidence_sql=f"SELECT sum({num(c)}) AS total FROM left_data UNION ALL SELECT sum({num(c)}) FROM right_data"))
    facts.sort(key=lambda f: (0 if f.severity == "warning" else 1 if f.severity == "notable" else 2, -f.score))
    for i, f in enumerate(facts, 1):
        f.id = f"R{i:03d}"
    kl, kr = con.execute(f"{ctes} SELECT count(ln), count(rn) FROM J").fetchone()
    summary = {"left": ln, "right": rn, "rows_left": int(rows_l), "rows_right": int(rows_r), "keys_left": int(kl), "keys_right": int(kr), "matched": match,
               "missing_in_right": miss, "extra_in_right": extra, "mismatched": mism,
               "match_rate": match / max(kl, 1), "tolerance": tol, "relative_tolerance": rel_tol, "key": ", ".join(key), "compare": ", ".join(compare)}
    r = SubReport(kind="reconcile", title=f"Reconciliation: {ln} vs {rn}", summary=summary, facts=facts,
                  tables={"Exceptions": (exc_cols, exc_rows),
                          "Bridge": (["column", f"total {ln}", f"total {rn}", "difference", f"missing in {rn}", f"extra in {rn}", "mismatches", "within-tolerance residue", "check (should be ~0)"], bridge_rows)},
                  setup_sql=[a.source_sql(), b.source_sql(),
                             "CREATE VIEW left_data AS SELECT * FROM left_raw", "CREATE VIEW right_data AS SELECT * FROM right_raw"])
    if len(exc_rows) >= cfg.recon_max_exceptions:
        r.warnings.append(f"Exception list truncated to {cfg.recon_max_exceptions:,} rows; counts above are exact.")
    if any(v[0] for v in dups.values()):
        r.warnings.append("Duplicate keys were aggregated (amounts summed) before matching.")
    r.provenance = provenance.build([provenance.input_record(a, "left", cfg.hash_data, progress), provenance.input_record(b, "right", cfg.hash_data, progress)],
                                    {"key": key, "compare": compare, "tolerance": tol, "relative_tolerance": rel_tol}, cfg, None, None, facts, command)
    progress("Done", 100)
    return r
