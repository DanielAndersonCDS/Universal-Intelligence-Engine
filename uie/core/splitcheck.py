"""Train/test overlap: exact and near-duplicate rows across splits, shared entities, and temporal overlap."""
from __future__ import annotations
from . import provenance
from .config import Config
from .facts import Fact
from .ingest import DataSource, open_pair
from .profile import compute_profile
from .semantics import Semantics, _lit
from .subreport import SubReport
from .util import qident


def _row_key(cols, kinds, decimals, near):
    parts = []
    for c in cols:
        q = qident(c)
        if near and kinds[c] == "numeric":
            parts.append(f"coalesce(CAST(round(CAST({q} AS DOUBLE), {decimals}) AS VARCHAR), '∅')")
        elif near and kinds[c] == "text":
            parts.append(f"coalesce(lower(trim(regexp_replace(CAST({q} AS VARCHAR), '\\s+', ' ', 'g'))), '∅')")
        else:
            parts.append(f"coalesce(CAST({q} AS VARCHAR), '∅')")
    return "md5(concat_ws('␟', " + ", ".join(parts) + "))"


def split_check(train, test=None, split_col=None, train_value=None, test_value=None, entity=None, ignore=None, label=None, time_col=None,
                cfg: Config | None = None, sem: Semantics | None = None, table_left=None, table_right=None,
                progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    entity = [entity] if isinstance(entity, str) else list(entity or ([sem.get("entity")] if sem and sem.get("entity") else []))
    progress("Opening datasets", 5)
    setup = []
    if test is None:
        if not (split_col and train_value is not None and test_value is not None):
            raise ValueError("Give a second dataset, or --split-col with --train-value and --test-value.")
        a = DataSource(view="raw").open(train, table_left)
        con, b = a.con, a
        con.execute(f"CREATE VIEW left_data AS SELECT * FROM raw WHERE {qident(split_col)} = {_lit(train_value)}")
        con.execute(f"CREATE VIEW right_data AS SELECT * FROM raw WHERE {qident(split_col)} = {_lit(test_value)}")
        setup = [a.source_sql(), f"CREATE VIEW left_data AS SELECT * FROM raw WHERE {qident(split_col)} = {_lit(train_value)}",
                 f"CREATE VIEW right_data AS SELECT * FROM raw WHERE {qident(split_col)} = {_lit(test_value)}"]
        ln, rn, inputs = f"{a.name}[{split_col}={train_value}]", f"{a.name}[{split_col}={test_value}]", [provenance.input_record(a, "data", cfg.hash_data, progress)]
        ignore = list(ignore or []) + [split_col]
    else:
        con, a, b = open_pair(train, test, table_left, table_right)
        con.execute("CREATE VIEW left_data AS SELECT * FROM left_raw")
        con.execute("CREATE VIEW right_data AS SELECT * FROM right_raw")
        setup = [a.source_sql(), b.source_sql(), "CREATE VIEW left_data AS SELECT * FROM left_raw", "CREATE VIEW right_data AS SELECT * FROM right_raw"]
        ln, rn = a.name, b.name
        inputs = [provenance.input_record(a, "train", cfg.hash_data, progress), provenance.input_record(b, "test", cfg.hash_data, progress)]
    pl, pr = compute_profile(con.cursor(), "left_data"), compute_profile(con.cursor(), "right_data")
    cl, cr = {c.name: c for c in pl.columns}, {c.name: c for c in pr.columns}
    skip = set(ignore or []) | ({label} if label else set()) | set(sem.cols_with_role("id", False) if sem else [])
    feats = []
    for n in cl:
        if n in cr and n not in skip and cl[n].kind in ("numeric", "text", "boolean", "temporal"):
            if pl.rows >= 500 and cl[n].distinct >= 0.97 * pl.rows:      # identifiers make every row unique
                continue
            feats.append(n)
    kinds = {n: cl[n].kind for n in feats}
    facts, table = [], []
    progress("Comparing rows", 35)
    if feats and pl.rows and pr.rows:
        for near in (False, True):
            k = _row_key(feats, kinds, cfg.near_round_decimals, near)
            sql = f"SELECT count(*) FROM (SELECT {k} AS k FROM right_data) WHERE k IN (SELECT {k} FROM left_data)"
            val = int(con.execute(sql).fetchone()[0])
            if near:
                near_n = val
                near_sql = sql
            else:
                exact_n = val
                exact_sql = sql
        for kind, n_, sql in (("exact", exact_n, exact_sql), ("near", near_n - exact_n, near_sql)):
            share = n_ / pr.rows
            table.append([kind, n_, share])
            if n_ > 0:
                facts.append(Fact(type="split_overlap", title=f"{kind.capitalize()} duplicate rows across splits", columns=feats,
                                  metrics={"kind": kind, "rows": n_, "share": share, "side": "test", "ref_side": "training", "columns": len(feats)},
                                  severity="warning" if share >= 0.01 or kind == "exact" and n_ >= 10 else "notable", score=min(1.0, share * 20 + 0.2),
                                  evidence_sql=sql))
    for e in entity:
        if e not in cl or e not in cr:
            continue
        q = qident(e)
        tot, shared, rows_shared = con.execute(
            f"SELECT count(DISTINCT {q}), count(DISTINCT {q}) FILTER (WHERE {q} IN (SELECT {q} FROM left_data WHERE {q} IS NOT NULL)), "
            f"count(*) FILTER (WHERE {q} IN (SELECT {q} FROM left_data WHERE {q} IS NOT NULL)) FROM right_data WHERE {q} IS NOT NULL").fetchone()
        if tot and shared:
            share, rshare = shared / tot, rows_shared / pr.rows
            table.append([f"entity:{e}", shared, share])
            facts.append(Fact(type="split_entities", title=f"Shared {e} values across splits", columns=[e],
                              metrics={"entity": e, "shared": int(shared), "entities": int(tot), "share": share, "row_share": rshare, "side": "test", "ref_side": "training"},
                              severity="warning" if share >= 0.2 else "notable", score=min(1.0, share + 0.2),
                              evidence_sql=f"SELECT count(DISTINCT {q}) AS test_entities, count(DISTINCT {q}) FILTER (WHERE {q} IN (SELECT {q} FROM left_data)) AS also_in_train FROM right_data WHERE {q} IS NOT NULL"))
    tc = time_col or (sem.get("time") if sem else None)
    if not tc:
        both = [n for n in cl if n in cr and cl[n].kind == "temporal" and cr[n].kind == "temporal"]
        tc = both[0] if both else None
    if tc and tc in cl and tc in cr and cl[tc].max is not None and cr[tc].min is not None and cr[tc].min < cl[tc].max:
        facts.append(Fact(type="split_time", title=f"Time overlap in {tc}", columns=[tc], severity="notable", score=0.5,
                          metrics={"message": f"The test period starts at {cr[tc].min} but the training period runs until {cl[tc].max}; the splits overlap in `{tc}`, "
                                              f"so a time-ordered evaluation would leak the future into training."},
                          evidence_sql=f"SELECT (SELECT max({qident(tc)}) FROM left_data) AS train_max, (SELECT min({qident(tc)}) FROM right_data) AS test_min"))
    facts.sort(key=lambda f: (0 if f.severity == "warning" else 1, -f.score))
    for i, f in enumerate(facts, 1):
        f.id = f"S{i:03d}"
    summary = {"train": ln, "test": rn, "rows_train": pl.rows, "rows_test": pr.rows, "columns_compared": len(feats),
               "excluded_columns": ", ".join(sorted(skip & set(cl))) or "none", "entities_checked": ", ".join(entity) or "none"}
    r = SubReport(kind="splits", title=f"Train/test overlap: {ln} vs {rn}", summary=summary, facts=facts,
                  tables={"Overlap": (["check", "test rows / values", "share"], table)}, setup_sql=setup)
    r.provenance = provenance.build(inputs, {"entity": entity, "ignore": sorted(skip), "label": label, "time_col": tc, "split_col": split_col,
                                             "near_round_decimals": cfg.near_round_decimals}, cfg, sem if sem and (sem.columns or sem.dataset) else None, None, facts, command)
    progress("Done", 100)
    return r
