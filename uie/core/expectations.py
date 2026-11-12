"""Export an expectations (schema + profile) file from a dataset and validate later datasets against it."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
from . import provenance, stats
from .config import Config
from .facts import Fact
from .ingest import DataSource
from .profile import compute_profile
from .semantics import Semantics, build_clean_view
from .subreport import SubReport
from .util import qident

VERSION = 1


def create(engine, cfg: Config | None = None, max_categories: int = 50) -> dict:
    """Build expectations from the dataset currently open in `engine` (after semantics cleaning)."""
    cfg = cfg or engine.cfg
    con = engine.src.con.cursor()
    prof = compute_profile(con, "data")
    cols = {}
    num = [c for c in prof.columns if c.kind == "numeric" and c.std]
    cum = {}
    if num:
        edges = {}
        for c in num:
            r = con.execute(f"SELECT quantile_cont({qident(c.name)}, {[i / 10 for i in range(1, 10)]!r}) FROM data").fetchone()[0]
            edges[c.name] = sorted(set(float(e) for e in r if e is not None))
        exprs = []
        for c in num:
            exprs += [f"count_if({qident(c.name)} <= {e!r})" for e in edges[c.name]] + [f"count({qident(c.name)})"]
        row = con.execute("SELECT " + ", ".join(exprs) + " FROM data").fetchone()
        pos = 0
        for c in num:
            k = len(edges[c.name]) + 1
            vals = [int(v) for v in row[pos:pos + k]]
            pos += k
            tot = vals[-1] or 1
            cumv = vals[:-1] + [tot]
            props = [cumv[0] / tot] + [(cumv[i] - cumv[i - 1]) / tot for i in range(1, len(cumv))]
            cum[c.name] = {"edges": edges[c.name], "proportions": props}
    for c in prof.columns:
        e = {"dtype": c.dtype, "kind": c.kind, "max_null_frac": round(min(1.0, (c.nulls / max(prof.rows, 1)) * 1.5 + 0.01), 4),
             "observed_null_frac": round(c.nulls / max(prof.rows, 1), 6)}
        if prof.rows >= 500 and c.distinct >= 0.97 * prof.rows and c.kind in ("text", "numeric"):
            e["unique"] = True
        if c.kind == "numeric" and c.min is not None:
            pad = 0.1 * (c.max - c.min)
            e.update(min=c.min, max=c.max, lo=c.min - pad, hi=c.max + pad)
            if c.name in cum:
                e["distribution"] = cum[c.name]
        elif c.kind in ("text", "boolean") and 0 < c.distinct <= max_categories:
            rs = con.execute(f"SELECT CAST({qident(c.name)} AS VARCHAR), count(*) FROM data WHERE {qident(c.name)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC").fetchall()
            tot = sum(r[1] for r in rs) or 1
            e["allowed_values"] = [r[0] for r in rs]
            e["proportions"] = {r[0]: r[1] / tot for r in rs}
        elif c.kind == "temporal" and c.min is not None:
            e.update(min=str(c.min), max=str(c.max))
        cols[c.name] = e
    return {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": __import__("uie").__version__,
            "source": provenance.input_record(engine.src, "data", cfg.hash_data) | {"rows": prof.rows},
            "semantics_digest": engine.sem.digest() if (engine.sem.columns or engine.sem.dataset) else None,
            "rows": {"observed": prof.rows, "min": int(prof.rows * 0.5), "max": int(prof.rows * 2)},
            "psi_warning": cfg.psi_warning, "psi_notable": cfg.psi_notable, "columns": cols}


def save(exp: dict, path):
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        p.write_text(yaml.safe_dump(exp, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        p.write_text(json.dumps(exp, indent=2, default=str), encoding="utf-8")


def load(path) -> dict:
    from pathlib import Path
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    exp = json.loads(text) if p.suffix.lower() == ".json" else __import__("yaml").safe_load(text)
    if not isinstance(exp, dict) or "columns" not in exp:
        raise ValueError("Not an expectations file (missing 'columns').")
    return exp


def validate(path, exp: dict, sem: Semantics | None = None, cfg: Config | None = None, table=None,
             progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    src = DataSource(view="raw").open(path, table)
    con = src.con
    build_clean_view(con, "raw", "data", sem, qident)
    prof = compute_profile(con.cursor(), "data")
    have = {c.name: c for c in prof.columns}
    facts = []
    pw, pn = exp.get("psi_warning", cfg.psi_warning), exp.get("psi_notable", cfg.psi_notable)

    def add(type_, title, cols, msg, sev, score, sql=""):
        facts.append(Fact(type=type_, title=title, columns=cols, metrics={"message": msg}, severity=sev, score=score, evidence_sql=sql))

    r = exp.get("rows") or {}
    if r and not (r.get("min", 0) <= prof.rows <= r.get("max", float("inf"))):
        add("expect_rows", "Row count outside expected range", [], f"The dataset has {prof.rows:,} rows; expected between {r['min']:,} and {r['max']:,}.",
            "warning", 0.9, "SELECT count(*) AS rows FROM data")
    progress("Checking schema", 20)
    for n, e in exp["columns"].items():
        if n not in have:
            add("expect_schema", f"Missing column {n}", [n], f"Column `{n}` is expected but missing.", "warning", 0.95)
            continue
        c = have[n]
        if c.kind != e["kind"]:
            add("expect_schema", f"Type changed: {n}", [n], f"`{n}` is now {c.dtype} ({c.kind}); expected {e['dtype']} ({e['kind']}).", "warning", 0.9)
    for n in have:
        if n not in exp["columns"]:
            add("expect_schema", f"New column {n}", [n], f"Column `{n}` is not in the expectations file.", "notable", 0.5)
    progress("Checking values", 45)
    for n, e in exp["columns"].items():
        c = have.get(n)
        if not c or c.kind != e["kind"]:
            continue
        q = qident(n)
        frac = c.nulls / max(prof.rows, 1)
        if frac > e["max_null_frac"] + 1e-9:
            add("expect_nulls", f"Missing values in {n}", [n], f"`{n}` is {frac * 100:.1f}% missing; the limit is {e['max_null_frac'] * 100:.1f}% (observed {e['observed_null_frac'] * 100:.1f}% at export).",
                "notable", frac, f"SELECT count(*) FILTER (WHERE {q} IS NULL) AS missing, count(*) AS total FROM data")
        if e.get("unique"):
            d = con.execute(f"SELECT count({q}) - count(DISTINCT {q}) FROM data").fetchone()[0]
            if d:
                add("expect_unique", f"Duplicate values in {n}", [n], f"`{n}` was unique at export but now has {int(d):,} repeated values.", "warning", 0.8,
                    f"SELECT {q}, count(*) AS copies FROM data GROUP BY 1 HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 100")
        if c.kind == "numeric" and "lo" in e:
            out, nn = con.execute(f"SELECT count_if({q} < {e['lo']!r} OR {q} > {e['hi']!r}), count({q}) FROM data").fetchone()
            if out:
                sh = out / max(nn, 1)
                add("expect_range", f"Out-of-range values in {n}", [n], f"{int(out):,} values ({sh * 100:.2f}%) in `{n}` fall outside [{e['lo']:.6g}, {e['hi']:.6g}] (observed range at export: {e['min']:.6g} to {e['max']:.6g}).",
                    "notable" if sh >= 0.001 else "info", min(1.0, sh * 20),
                    f"SELECT * FROM data WHERE {q} < {e['lo']!r} OR {q} > {e['hi']!r} LIMIT 100")
            if "distribution" in e and len(e["distribution"]["edges"]) >= 2:
                ed = e["distribution"]["edges"]
                row = con.execute("SELECT " + ", ".join(f"count_if({q} <= {x!r})" for x in ed) + f", count({q}) FROM data").fetchone()
                cum = [int(v) for v in row[:-1]] + [int(row[-1])]
                cur = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
                ref = [p * 1000 for p in e["distribution"]["proportions"]]
                psi = stats.psi(ref, cur)
                if psi >= pn:
                    add("expect_drift", f"Distribution drift in {n}", [n], f"`{n}` has drifted from its exported distribution (PSI = {psi:.3f}).",
                        "warning" if psi >= pw else "notable", min(1.0, psi))
        elif c.kind in ("text", "boolean") and "allowed_values" in e:
            allowed = ", ".join("'" + v.replace("'", "''") + "'" for v in e["allowed_values"])
            unseen = con.execute(f"SELECT CAST({q} AS VARCHAR), count(*) FROM data WHERE {q} IS NOT NULL AND CAST({q} AS VARCHAR) NOT IN ({allowed}) GROUP BY 1 ORDER BY 2 DESC").fetchall()
            if unseen:
                tot = sum(u[1] for u in unseen)
                ex = ", ".join(f"{u[0]} ({u[1]:,})" for u in unseen[:5])
                add("expect_category", f"Unseen categories in {n}", [n], f"`{n}` has {len(unseen):,} unseen value(s) covering {tot:,} rows: {ex}.", "notable", min(1.0, tot / max(prof.rows, 1) * 10 + 0.2),
                    f"SELECT CAST({q} AS VARCHAR) AS value, count(*) AS rows FROM data WHERE {q} IS NOT NULL AND CAST({q} AS VARCHAR) NOT IN ({allowed}) GROUP BY 1 ORDER BY 2 DESC")
            ref = e.get("proportions") or {}
            if ref and prof.rows:
                cur = dict(con.execute(f"SELECT CAST({q} AS VARCHAR), count(*) FROM data WHERE {q} IS NOT NULL GROUP BY 1").fetchall())
                keys = sorted(set(ref) | set(cur))
                psi = stats.psi([ref.get(k, 0) * 1000 for k in keys], [cur.get(k, 0) for k in keys])
                if psi >= pn:
                    add("expect_drift", f"Category mix drift in {n}", [n], f"The category mix of `{n}` has drifted from the exported mix (PSI = {psi:.3f}).",
                        "warning" if psi >= pw else "notable", min(1.0, psi))
    facts.sort(key=lambda f: (0 if f.severity == "warning" else 1 if f.severity == "notable" else 2, -f.score))
    for i, f in enumerate(facts, 1):
        f.id = f"E{i:03d}"
    summary = {"dataset": src.name, "rows": prof.rows, "columns": len(have), "expectations_created": exp.get("created_utc"),
               "expectations_source": (exp.get("source") or {}).get("name"), "violations": len(facts)}
    rep = SubReport(kind="validate", title=f"Validation: {src.name}", summary=summary, facts=facts, setup_sql=[src.source_sql()])
    rep.provenance = provenance.build([provenance.input_record(src, "data", cfg.hash_data, progress)], {"expectations_created": exp.get("created_utc")},
                                      cfg, sem if sem and (sem.columns or sem.dataset) else None, None, facts, command)
    progress("Done", 100)
    return rep
