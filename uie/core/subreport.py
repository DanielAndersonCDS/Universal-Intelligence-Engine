"""Result container for the multi-dataset tools (reconcile, drift, split check, validate)."""
from __future__ import annotations
import csv, io, json
from dataclasses import dataclass, field
from .facts import SEV_ORDER
from . import provenance


@dataclass
class SubReport:
    kind: str                       # reconcile | drift | splits | validate
    title: str
    summary: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)       # name -> (columns, rows)
    setup_sql: list = field(default_factory=list)    # statements that recreate the views the evidence SQL uses
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def worst(self):
        s = [f.severity for f in self.facts if f.active]
        return min(s, key=lambda x: SEV_ORDER[x]) if s else None

    def gate(self, fail_on: str) -> bool:
        """True when any active finding is at or above `fail_on` severity."""
        if fail_on == "never":
            return False
        lim = SEV_ORDER[fail_on]
        return any(SEV_ORDER[f.severity] <= lim for f in self.facts if f.active)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {"kind": self.kind, "title": self.title, "summary": self.summary, "provenance": self.provenance,
                "warnings": self.warnings, "setup_sql": self.setup_sql,
                "findings": [{**asdict(f), "text": f.narrate()} for f in self.facts],
                "tables": {k: {"columns": c, "rows": r} for k, (c, r) in self.tables.items()}}


def _cell(v):
    return "" if v is None else (f"{v:,.4g}" if isinstance(v, float) else str(v))


def render_markdown(r: SubReport, max_rows: int = 25) -> str:
    L = [f"# {r.title}", "", f"*Generated {r.provenance.get('created_utc', '')}.*", "", "## Summary", "",
         "| Metric | Value |", "|---|---|"]
    L += [f"| {k} | {v if not isinstance(v, float) else f'{v:,.4g}'} |" for k, v in r.summary.items()]
    L += ["", "## Findings", ""]
    L += [f"- **{f.id or f.severity.upper()}** [{f.severity.upper()}] {f.narrate()}" for f in r.facts if f.active] or ["- No findings."]
    L.append("")
    for name, (cols, rows) in r.tables.items():
        if not rows:
            continue
        L += [f"## {name}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        L += ["| " + " | ".join(_cell(v).replace("|", "\\|") for v in row) + " |" for row in rows[:max_rows]]
        if len(rows) > max_rows:
            L.append(f"\n*Showing {max_rows} of {len(rows):,} rows; the CSV export has all of them.*")
        L.append("")
    L += provenance.render_markdown(r.provenance)
    ev = [f for f in r.facts if f.evidence_sql]
    if ev or r.setup_sql:
        L += ["## Evidence appendix", ""]
        if r.setup_sql:
            L += ["Run these first (DuckDB) to recreate the views the queries use:", "", "```sql"] + r.setup_sql + ["```", ""]
        for f in ev:
            L += [f"### {f.id or f.type} {f.title}", "", "```sql", f.evidence_sql, "```", ""]
    if r.warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]
    return "\n".join(L)


def tables_csv(r: SubReport, table: str | None = None) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    if table is None:
        w.writerow(["id", "severity", "type", "title", "text", "evidence_sql"])
        for f in r.facts:
            w.writerow([f.id, f.severity, f.type, f.title, f.narrate(), f.evidence_sql])
    else:
        cols, rows = r.tables[table]
        w.writerow(cols)
        w.writerows(rows)
    return buf.getvalue()


def to_json(r: SubReport) -> str:
    return json.dumps(r.to_dict(), indent=2, default=str)
