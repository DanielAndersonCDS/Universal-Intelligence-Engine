from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from .profile import Profile

SEV_ORDER = {"warning": 0, "notable": 1, "info": 2}
PATTERN_TYPES = {"trend", "changepoint", "correlation", "near_duplicate", "concentration", "volatility", "vol_shift",
                 "return_correlation", "variance_driver", "group_contrast"}
ANOMALY_TYPES = {"outliers", "quality_duplicates", "near_duplicate_tx", "threshold_bunching", "benford",
                 "round_numbers", "target_leakage", "leakage_missingness", "spurious_correlation", "time_gap"}
ACTIVE = ("new", "reopened")


@dataclass
class Fact:
    type: str
    title: str
    metrics: dict
    severity: str = "info"
    score: float = 0.0
    columns: list = field(default_factory=list)
    evidence_sql: str = ""
    chart: dict | None = None
    confidence: str = "descriptive_only"   # never claims causation
    id: str = ""
    group: dict | None = None              # {"column": ..., "value": ...} when found inside one group
    fingerprint: str = ""                  # stable identity across runs (see triage.py)
    magnitude: float = 0.0                 # how big the finding is, used to re-raise dismissed findings
    status: str = "new"                    # new | accepted | dismissed | reopened
    status_note: str = ""

    def narrate(self) -> str:
        from .narration import narrate
        return narrate(self)

    @property
    def is_quality(self) -> bool:
        return self.type.startswith("quality_")

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


@dataclass
class FactSheet:
    dataset: str
    path: str
    rows: int
    cols: int
    size_bytes: int
    seconds: float
    time_col: str | None
    profile: Profile
    facts: list
    warnings: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    narrative: str = ""
    narrator: str = "template"
    group_by: str | None = None
    label: str | None = None
    provenance: dict = field(default_factory=dict)
    semantics: dict | None = None

    def active(self):
        return [f for f in self.facts if f.active]

    def triaged(self):
        return [f for f in self.facts if not f.active]

    def patterns(self) -> int:
        return sum(1 for f in self.active() if f.type in PATTERN_TYPES)

    def anomalies(self) -> int:
        return sum(1 for f in self.active() if f.type in ANOMALY_TYPES
                   or (f.type == "quality_nulls" and f.severity == "warning"))

    def ranked(self):
        return [f for f in self.active() if not f.is_quality]

    def quality(self):
        return [f for f in self.active() if f.is_quality]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["facts"] = [{**asdict(f), "text": f.narrate()} for f in self.facts]
        return d
