from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .util import qident, to_float

NUMERIC = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
           "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE", "REAL"}


def kind_of(t: str) -> str:
    t = t.upper()
    if t in NUMERIC or t.startswith("DECIMAL"):
        return "numeric"
    if t == "DATE" or t.startswith("TIMESTAMP"):
        return "temporal"
    if t == "BOOLEAN":
        return "boolean"
    if t == "VARCHAR":
        return "text"
    return "other"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    nulls: int = 0
    distinct: int = 0
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    top: list = field(default_factory=list)


@dataclass
class Profile:
    rows: int
    columns: list

    def col(self, name):
        for c in self.columns:
            if c.name == name:
                return c
        return None


def compute_profile(con, rel: str, progress=None) -> Profile:
    desc = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    cols = [ColumnProfile(r[0], str(r[1]), kind_of(str(r[1]))) for r in desc]
    rows = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    CH = 40
    for i in range(0, len(cols), CH):
        chunk = cols[i:i + CH]
        exprs, layout = [], []
        for c in chunk:
            q = qident(c.name)
            parts = [f"count(*) - count({q})"]
            if c.kind != "other":
                parts.append(f"approx_count_distinct({q})")
            if c.kind == "numeric":
                parts += [f"min({q})", f"max({q})", f"avg({q})", f"stddev_samp({q})",
                          f"approx_quantile({q}, 0.25)", f"approx_quantile({q}, 0.5)",
                          f"approx_quantile({q}, 0.75)"]
            elif c.kind == "temporal":
                parts += [f"min({q})", f"max({q})"]
            exprs += parts
            layout.append(len(parts))
        row = con.execute(f"SELECT {', '.join(exprs)} FROM {rel}").fetchone()
        pos = 0
        for c, n in zip(chunk, layout):
            v = row[pos:pos + n]
            pos += n
            c.nulls = int(v[0] or 0)
            if c.kind != "other":
                c.distinct = int(v[1] or 0)
            if c.kind == "numeric":
                c.min, c.max = to_float(v[2]), to_float(v[3])
                c.mean, c.std = to_float(v[4]), to_float(v[5])
                c.q25, c.median, c.q75 = to_float(v[6]), to_float(v[7]), to_float(v[8])
            elif c.kind == "temporal":
                c.min, c.max = v[2], v[3]
        if progress:
            progress("Profiling columns", 5 + int(20 * min(i + CH, len(cols)) / max(1, len(cols))))
    for c in [c for c in cols if c.kind in ("text", "boolean")][:30]:
        q = qident(c.name)
        try:
            c.top = [(str(v), int(n)) for v, n in con.execute(
                f"SELECT {q}, count(*) n FROM {rel} WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
            ).fetchall()]
        except Exception:
            pass
    return Profile(rows=int(rows), columns=cols)
