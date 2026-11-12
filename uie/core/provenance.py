"""Provenance manifest: what data, which tool version, which parameters, when. Embedded in every export."""
from __future__ import annotations
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_files(files, base=None) -> str:
    """One digest for a set of files (folder datasets): hash of sorted 'relative-path NUL file-hash' lines."""
    h = hashlib.sha256()
    base = Path(base) if base else None
    for f in sorted(files):
        rel = str(Path(f).relative_to(base)) if base else str(f)
        h.update(rel.replace("\\", "/").encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


def input_record(src, role="data", compute_hash=True, progress=None) -> dict:
    rec = {"role": role, "path": str(Path(src.path).resolve()) if src.path else None, "name": src.name,
           "size_bytes": src.size_bytes, "table": src.table, "files": len(src.files), "sha256": None}
    if compute_hash and src.files:
        if progress:
            progress(f"Hashing {src.name}", 2)
        try:
            p = Path(src.path)
            rec["sha256"] = sha256_file(src.files[0]) if not p.is_dir() else sha256_files(src.files, p)
        except OSError as e:
            rec["sha256_error"] = str(e)
    return rec


def runtime() -> dict:
    import duckdb, numpy
    return {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "numpy": numpy.__version__,
            "platform": platform.platform()}


def _round(o):
    if isinstance(o, float):
        return float(f"{o:.6g}")
    if isinstance(o, dict):
        return {k: _round(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    return o


def result_digest(facts) -> str:
    """SHA-256 over the findings (type, fingerprint, metrics rounded to 6 significant digits, evidence SQL).
    Approximate aggregates (approx_quantile etc.) can differ in the last digits between runs; rounding absorbs that."""
    blob = json.dumps([[f.type, f.fingerprint, _round(f.metrics), f.evidence_sql] for f in facts],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def build(inputs, params, cfg=None, sem=None, triage=None, facts=None, command=None) -> dict:
    from .. import __version__
    m = {"tool": {"name": "UIE Universal Intelligence Engine", "version": __version__},
         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "runtime": runtime(), "inputs": inputs, "parameters": params,
         "config": cfg.to_dict() if cfg else None,
         "semantics": ({"path": sem.source_path, "digest": sem.digest(), "inferred_roles": sem.inferred} if sem else None)}
    if command:
        m["command"] = command
    if triage is not None:
        t = {"path": str(triage.path), "entries": len(triage.entries)}
        try:
            t["sha256"] = sha256_file(triage.path) if triage.path.is_file() else None
        except OSError:
            pass
        m["triage"] = t
    if facts is not None:
        m["result_digest"] = result_digest(facts)
        m["findings"] = len(facts)
    return m


def render_markdown(prov: dict) -> list:
    if not prov:
        return []
    L = ["## Provenance", "", "| Item | Value |", "|---|---|",
         f"| Tool | {prov['tool']['name']} v{prov['tool']['version']} |", f"| Created (UTC) | {prov['created_utc']} |"]
    for i in prov.get("inputs", []):
        L.append(f"| Input ({i['role']}) | {i['name']} ({i['size_bytes']:,} bytes) |")
        L.append(f"| SHA-256 ({i['role']}) | `{i.get('sha256') or 'not computed'}` |")
    rt = prov.get("runtime", {})
    L.append(f"| Runtime | Python {rt.get('python')}, DuckDB {rt.get('duckdb')}, NumPy {rt.get('numpy')} |")
    if prov.get("semantics"):
        s = prov["semantics"]
        L.append(f"| Semantics | {s.get('path') or 'inferred only'} (digest `{s['digest'][:16]}`) |")
    if prov.get("triage"):
        L.append(f"| Triage store | {prov['triage']['entries']} reviewed findings |")
    if prov.get("result_digest"):
        L.append(f"| Result digest | `{prov['result_digest']}` |")
    p = {k: v for k, v in (prov.get("parameters") or {}).items() if v not in (None, "", [], {})}
    if p:
        L.append(f"| Parameters | `{json.dumps(p, default=str)}` |")
    return L + [""]
