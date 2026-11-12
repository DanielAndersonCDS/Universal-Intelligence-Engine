"""Persistent dismiss / accept for findings. Reviewed findings stay out of repeat runs (and CI gates) unless they get worse."""
from __future__ import annotations
import getpass, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("accepted", "dismissed")
_KEYS = {
    "outliers": ["column"], "correlation": ["a", "b"], "near_duplicate": ["a", "b"], "trend": ["label"], "trend_caveat": ["label"],
    "changepoint": ["label"], "concentration": ["column", "group_column", "group_value"], "quality_nulls": ["column"],
    "quality_constant": ["column"], "quality_id_like": ["column"], "quality_text_numeric": ["column"], "quality_sentinel": ["column"],
    "quality_derived": ["column"], "volatility": ["label"], "vol_shift": ["label"], "return_correlation": ["a", "b"],
    "spurious_correlation": ["a", "b"], "benford": ["column"], "round_numbers": ["column", "modulus"],
    "threshold_bunching": ["column", "threshold"], "near_duplicate_tx": ["amount", "counterparty"],
    "variance_driver": ["measure", "basis", "dimension"], "time_gap": ["column"], "target_leakage": ["feature", "label"],
    "leakage_missingness": ["feature", "label"], "class_imbalance": ["label"], "near_constant": ["column"],
    "high_cardinality": ["column"], "group_contrast": ["column", "group_column"], "drift": ["column"],
    "drift_schema": ["column", "change"], "split_overlap": ["kind", "side"], "split_entities": ["entity", "side"],
}
_MAG = {"outliers": "share", "correlation": lambda m: abs(m["r"]), "near_duplicate": lambda m: abs(m["r"]), "trend": "pct",
        "changepoint": "pct", "concentration": "value_share", "quality_nulls": "frac", "quality_duplicates": "frac",
        "benford": "mad", "round_numbers": "share", "near_duplicate_tx": "share", "target_leakage": "purity",
        "leakage_missingness": "purity", "volatility": "ann_vol", "vol_shift": "ratio", "group_contrast": "eta2",
        "drift": "psi", "variance_driver": lambda m: abs(m["pct"]), "spurious_correlation": lambda m: abs(m["level_r"]),
        "return_correlation": lambda m: abs(m["r"]), "class_imbalance": lambda m: 1 - m["minority_share"],
        "threshold_bunching": lambda m: m["below"] / (m["above"] + 1), "split_overlap": "share", "split_entities": "share"}


def _num(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def fingerprint(f) -> str:
    keys = _KEYS.get(f.type, [])
    ident = [f.type, sorted(f.columns),
             f.group or None, [(k, _num(f.metrics.get(k)) if isinstance(f.metrics.get(k), (int, float)) and k in ("modulus", "threshold")
                                else str(f.metrics.get(k))) for k in keys]]
    return hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def magnitude(f) -> float:
    spec = _MAG.get(f.type)
    try:
        v = spec(f.metrics) if callable(spec) else (f.metrics[spec] if spec else f.score)
        return float(v)
    except Exception:
        return float(f.score or 0.0)


def assign(facts):
    seen = {}
    for f in facts:
        fp = fingerprint(f)
        n = seen.get(fp, 0)
        seen[fp] = n + 1
        f.fingerprint = fp if n == 0 else f"{fp}-{n}"
        f.magnitude = magnitude(f)


class TriageStore:
    def __init__(self, path):
        self.path = Path(path)
        self.entries: dict = {}
        self.load()

    @staticmethod
    def default_path(dataset_path) -> Path:
        p = Path(dataset_path)
        side = p.with_name(p.name + ".uie-triage.json") if not p.is_dir() else p / ".uie-triage.json"
        try:
            if os.access(side.parent, os.W_OK):
                return side
        except OSError:
            pass
        home = Path.home() / ".uie" / "triage"
        return home / (hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16] + ".json")

    def load(self):
        if self.path.is_file():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8")).get("entries", {})
            except (OSError, ValueError):
                self.entries = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"version": 1, "entries": self.entries}, indent=2, sort_keys=True), encoding="utf-8")

    def set(self, fact, status: str, note: str = ""):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        self.entries[fact.fingerprint] = {
            "status": status, "note": note, "type": fact.type, "title": fact.title, "magnitude": fact.magnitude,
            "text": fact.narrate()[:300], "by": _user(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        fact.status, fact.status_note = status, note
        self.save()

    def set_fingerprint(self, prefix: str, status: str, note: str = "") -> str:
        hits = [k for k in self.entries if k.startswith(prefix)]
        if len(hits) != 1:
            raise KeyError(f"fingerprint '{prefix}' matches {len(hits)} stored entries")
        self.entries[hits[0]].update(status=status, note=note, by=_user(), at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.save()
        return hits[0]

    def remove(self, key: str):
        hits = [k for k in self.entries if k.startswith(key)]
        for k in hits:
            del self.entries[k]
        if hits:
            self.save()
        return len(hits)

    def restore(self, fact):
        if self.entries.pop(fact.fingerprint, None) is not None:
            self.save()
        fact.status, fact.status_note = "new", ""

    def apply(self, facts, reraise_factor: float = 1.5) -> dict:
        counts = {"accepted": 0, "dismissed": 0, "reopened": 0}
        for f in facts:
            e = self.entries.get(f.fingerprint)
            if not e:
                continue
            old = float(e.get("magnitude") or 0)
            worse = f.magnitude > max(old * reraise_factor, old + 1e-9) and f.magnitude - old > 1e-6
            if worse and e["status"] in STATUSES:
                f.status = "reopened"
                f.status_note = f"previously {e['status']} at {old:.4g}; now {f.magnitude:.4g}"
                counts["reopened"] += 1
            else:
                f.status, f.status_note = e["status"], e.get("note", "")
                counts[e["status"]] += 1
        return counts


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
