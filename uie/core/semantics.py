"""Column semantics: what the columns *mean*, supplied by the user (YAML/JSON) or inferred from names.

    dataset:
      time: Date               # time column
      group_by: Sex            # run detectors within each group
      entity: Name             # who a row belongs to (ticker, account, customer)
      label: Target            # ML label (enables leakage / imbalance checks)
      amount: Amount           # monetary column(s)
      counterparty: Vendor
      key: [TransactionID]     # default reconciliation key
      thresholds: [10000]      # approval / reporting limits to test for bunching just below
      dedup_window_days: 3
    columns:
      Squat4Kg:
        negative_is_missing: true       # negative = failed attempt
        note: "negative = failed attempt"
        unit: kg
      TotalKg:
        role: derived
        derived_from: [BestSquatKg, BestBenchKg, BestDeadliftKg]
      Close: {role: price, currency: USD}
      Notes: {role: ignore}
      Wilks: {exclude: [outliers]}
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, field
from pathlib import Path

ROLES = {"measure", "id", "derived", "label", "entity", "time", "amount", "counterparty", "price", "category", "ignore"}
DETECTORS = {"outliers", "correlation", "trends", "segments", "benford", "rounding", "returns", "leakage", "all"}
# roles that keep a column out of numeric detectors
_ROLE_EXCLUDES = {"id": {"all"}, "ignore": {"all"}, "entity": {"all"}, "time": {"all"}, "category": {"all"},
                  "label": {"outliers", "trends", "segments"}, "derived": {"correlation"}}


@dataclass
class ColumnSem:
    name: str
    role: str | None = None
    sentinels: list = field(default_factory=list)
    negative_is_missing: bool = False
    note: str = ""
    unit: str | None = None
    currency: str | None = None
    exclude: set = field(default_factory=set)
    derived_from: list = field(default_factory=list)

    def rule_text(self) -> str:
        bits = []
        if self.negative_is_missing:
            bits.append("negative values")
        if self.sentinels:
            bits.append("values " + ", ".join(map(str, self.sentinels)))
        return (" and ".join(bits) + " treated as missing") + (f" ({self.note})" if self.note else "")


@dataclass
class Semantics:
    columns: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    source_path: str | None = None
    inferred: dict = field(default_factory=dict)       # role -> [columns] guessed from names
    _raw: dict = field(default_factory=dict)

    # ---- loading ----
    @classmethod
    def from_dict(cls, d: dict, source_path=None) -> "Semantics":
        d = d or {}
        s = cls(dataset=dict(d.get("dataset") or {}), source_path=source_path, _raw=d)
        for name, spec in (d.get("columns") or {}).items():
            spec = spec or {}
            role = spec.get("role")
            if role is not None and role not in ROLES:
                raise ValueError(f"column '{name}': unknown role '{role}' (valid: {', '.join(sorted(ROLES))})")
            ex = set(spec.get("exclude") or [])
            bad = ex - DETECTORS
            if bad:
                raise ValueError(f"column '{name}': unknown exclude {sorted(bad)} (valid: {', '.join(sorted(DETECTORS))})")
            s.columns[name] = ColumnSem(
                name=name, role=role, sentinels=list(spec.get("sentinels") or []),
                negative_is_missing=bool(spec.get("negative_is_missing", False)), note=str(spec.get("note") or ""),
                unit=spec.get("unit"), currency=spec.get("currency"), exclude=ex,
                derived_from=list(spec.get("derived_from") or []))
            if s.columns[name].derived_from and role is None:
                s.columns[name].role = "derived"
        ds = s.dataset
        for key, role in (("time", "time"), ("entity", "entity"), ("label", "label"), ("counterparty", "counterparty")):
            if ds.get(key):
                col = s._ensure(ds[key])
                col.role = col.role or role
        for key, role in (("amount", "amount"), ("price", "price")):
            v = ds.get(key)
            for n in ([v] if isinstance(v, str) else (v or [])):
                col = s._ensure(n)
                col.role = col.role or role
        return s

    @classmethod
    def from_file(cls, path) -> "Semantics":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            d = json.loads(text)
        else:
            import yaml
            d = yaml.safe_load(text)
        if d is not None and not isinstance(d, dict):
            raise ValueError("Semantics file must be a mapping with 'dataset' and/or 'columns' sections")
        return cls.from_dict(d or {}, str(p))

    @classmethod
    def sidecar_for(cls, data_path) -> "Semantics | None":
        p = Path(data_path)
        for cand in (p.with_name(p.name + ".uie.yaml"), p.with_suffix(".uie.yaml"), p.with_name(p.name + ".uie.yml"),
                     p.with_name(p.name + ".uie.json")):
            if cand.is_file():
                return cls.from_file(cand)
        return None

    def _ensure(self, name) -> ColumnSem:
        if name not in self.columns:
            self.columns[name] = ColumnSem(name=name)
        return self.columns[name]

    # ---- queries ----
    def role(self, name):
        c = self.columns.get(name)
        if c and c.role:
            return c.role
        for r, names in self.inferred.items():
            if name in names:
                return r
        return None

    def cols_with_role(self, role, include_inferred=True):
        out = [n for n, c in self.columns.items() if c.role == role]
        if include_inferred:
            out += [n for n in self.inferred.get(role, []) if n not in out and not (self.columns.get(n) and self.columns[n].role)]
        return out

    def excluded(self, name, detector) -> bool:
        c = self.columns.get(name)
        if c and (detector in c.exclude or "all" in c.exclude):
            return True
        r = self.role(name)
        return bool(r and (_ROLE_EXCLUDES.get(r, set()) & {"all", detector}))

    def derived_sources(self) -> dict:
        return {n: list(c.derived_from) for n, c in self.columns.items() if c.derived_from}

    def get(self, key, default=None):
        return self.dataset.get(key, default)

    def clean_rules(self) -> dict:
        return {n: c for n, c in self.columns.items() if c.negative_is_missing or c.sentinels}

    def validate(self, column_names) -> list:
        names = set(column_names)
        w = []
        for n, c in self.columns.items():
            if n not in names:
                w.append(f"Semantics refers to unknown column '{n}'.")
            for src in c.derived_from:
                if src not in names:
                    w.append(f"derived_from of '{n}' refers to unknown column '{src}'.")
        for k in ("time", "group_by", "entity", "label", "counterparty"):
            v = self.dataset.get(k)
            if v and v not in names:
                w.append(f"dataset.{k} refers to unknown column '{v}'.")
        for k in ("amount", "price", "key"):
            v = self.dataset.get(k)
            for n in ([v] if isinstance(v, str) else (v or [])):
                if n not in names:
                    w.append(f"dataset.{k} refers to unknown column '{n}'.")
        return w

    def digest(self) -> str:
        blob = json.dumps({"dataset": self.dataset, "raw_columns": self._raw.get("columns", {})}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"source": self.source_path, "dataset": self.dataset, "digest": self.digest(),
                "columns": {n: {k: v for k, v in {"role": c.role, "sentinels": c.sentinels or None,
                                                  "negative_is_missing": c.negative_is_missing or None,
                                                  "note": c.note or None, "unit": c.unit, "currency": c.currency,
                                                  "exclude": sorted(c.exclude) or None,
                                                  "derived_from": c.derived_from or None}.items() if v is not None}
                            for n, c in self.columns.items()},
                "inferred_roles": self.inferred}


_PATTERNS = {
    "amount": re.compile(r"(^|_)(amount|amt|payment|debit|credit|gross|net|invoice|txn_?value|transaction_?value)(_|$)|amount$", re.I),
    "counterparty": re.compile(r"vendor|payee|merchant|counterpart|supplier|beneficiary|payer|client", re.I),
    "price": re.compile(r"^(close|adj_?close|adjclose|px_?last|nav|settle|last_?price|price)$", re.I),
    "entity": re.compile(r"^(ticker|symbol|isin|cusip|account(_?id)?|customer_?id|user_?id|patient_?id|asset)$", re.I),
    "label": re.compile(r"^(label|target|is_?fraud|fraud|churn(ed)?|default(ed)?|outcome|y)$", re.I),
}


def infer_roles(profile, sem: Semantics, rows: int) -> dict:
    """Guess roles from column names. Explicit semantics always win. Returns {role: [cols]} (also stored on sem)."""
    out = {}
    for c in profile.columns:
        if sem.columns.get(c.name) and sem.columns[c.name].role:
            continue
        for role, rx in _PATTERNS.items():
            if not rx.search(c.name):
                continue
            ok = {"amount": c.kind == "numeric", "price": c.kind == "numeric" and (c.min or 0) > 0,
                  "counterparty": c.kind == "text" and 2 <= c.distinct < 0.9 * max(rows, 1),
                  "entity": 1 < c.distinct <= 5000 and c.kind in ("text", "numeric"),
                  "label": 2 <= c.distinct <= 50}[role]
            if ok:
                out.setdefault(role, []).append(c.name)
                break
    sem.inferred = out
    return out


def _lit(v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def clean_condition(r: "ColumnSem", q: str, numeric: bool, text: bool = False) -> str:
    """SQL condition that is true for cells the semantics say to treat as missing (empty string if none)."""
    parts = []
    if r.negative_is_missing and numeric:
        parts.append(f"{q} < 0")
    sv = [v for v in r.sentinels if (isinstance(v, (int, float)) and numeric) or (isinstance(v, str) and text)]
    if sv:
        parts.append(f"{q} IN ({', '.join(_lit(v) for v in sv)})")
    return " OR ".join(parts)


def build_clean_view(con, src_view: str, dst_view: str, sem: Semantics | None, qident) -> dict:
    """CREATE VIEW dst_view over src_view with sentinel/negative values replaced by NULL.
    Returns {column: number of cells nulled}."""
    desc = con.execute(f"DESCRIBE SELECT * FROM {src_view}").fetchall()
    rules = sem.clean_rules() if sem else {}
    select, counts = [], {}
    conds = {}
    for name, typ, *_ in desc:
        q, t = qident(name), str(typ).upper()
        r = rules.get(name)
        if r is None:
            select.append(q)
            continue
        numeric = t in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL"} or t.startswith("DECIMAL")
        cond = clean_condition(r, q, numeric, t == "VARCHAR")
        if not cond:
            select.append(q)
            continue
        conds[name] = cond
        select.append(f"CASE WHEN {cond} THEN NULL ELSE {q} END AS {q}")
    con.execute(f"CREATE OR REPLACE VIEW {dst_view} AS SELECT {', '.join(select)} FROM {src_view}")
    if conds:
        row = con.execute("SELECT " + ", ".join(f"count_if({c})" for c in conds.values()) + f" FROM {src_view}").fetchone()
        counts = {n: int(v or 0) for n, v in zip(conds, row)}
    return counts
