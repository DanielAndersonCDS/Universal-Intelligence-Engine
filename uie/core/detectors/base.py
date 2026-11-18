from __future__ import annotations
from dataclasses import dataclass, field
from ..config import Config
from ..profile import Profile, ColumnProfile
from ..semantics import Semantics


@dataclass
class Context:
    con: object
    rel: str
    profile: Profile
    time_col: str | None = None
    all_text: bool = False
    cfg: Config = field(default_factory=Config)
    sem: Semantics = field(default_factory=Semantics)
    label: str | None = None
    sentinel_counts: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)         # derived column -> [source columns] (auto-detected or declared)
    group: dict | None = None                           # set while running inside one group
    entity: str | None = None
    group_by: str | None = None
    notes: list = field(default_factory=list)

    def is_id_like(self, c: ColumnProfile) -> bool:
        if self.sem.role(c.name) == "id":
            return True
        r = self.profile.rows
        return (r >= 500 and c.distinct >= 0.97 * r
                and ("INT" in c.dtype.upper() or c.kind == "text"))

    def measures(self, detector: str | None = None):
        out = []
        for c in self.profile.columns:
            if not (c.kind == "numeric" and c.std and c.std > 0 and c.distinct > 1 and not self.is_id_like(c)):
                continue
            if detector and self.sem.excluded(c.name, detector):
                continue
            if self.sem.role(c.name) in ("id", "ignore", "entity", "time", "category"):
                continue
            if (self.entity and not self.group and self.sem.role(c.name) == "price"
                    and detector in ("outliers", "trends", "segments", "correlation")):
                continue                      # averaging prices across tickers is meaningless; use group_by to analyse each one
            out.append(c)
        return out

    def is_derived_pair(self, a: str, b: str) -> bool:
        return b in self.derived.get(a, ()) or a in self.derived.get(b, ())

    def roles(self, role: str):
        """Columns with a semantic role (declared first, then name-inferred), restricted to this dataset's columns."""
        have = {c.name for c in self.profile.columns}
        return [n for n in self.sem.cols_with_role(role) if n in have]

    def col(self, name):
        return self.profile.col(name)

    def sample_rows(self, cols, limit=50000):
        """Deterministic hash sample (same rows every run) of the given column names, as a list of tuples."""
        from ..util import qident
        names = [c.name for c in self.profile.columns][:12]
        mod = max(1, self.profile.rows // limit)
        where = f"WHERE hash({', '.join(qident(n) for n in names)}) % {mod} = 0" if mod > 1 else ""
        return self.con.execute(f"SELECT {', '.join(qident(c) for c in cols)} FROM {self.rel} {where}").fetchall()
