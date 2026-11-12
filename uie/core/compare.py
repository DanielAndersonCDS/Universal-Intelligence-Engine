from __future__ import annotations


def compare_profiles(a, b):
    """Descriptive drift between two datasets using their profiles only."""
    bm = {c.name: c for c in b.columns}
    am = {c.name: c for c in a.columns}
    out = []
    for ca in a.columns:
        cb = bm.get(ca.name)
        if cb is None:
            out.append({"column": ca.name, "flag": "REMOVED", "score": 9})
            continue
        row = {"column": ca.name, "flag": "", "score": 0.0}
        na, nb = ca.nulls / max(a.rows, 1), cb.nulls / max(b.rows, 1)
        row.update(null_a=na, null_b=nb)
        if ca.kind == "numeric" and cb.kind == "numeric" and ca.mean is not None and cb.mean is not None:
            row.update(mean_a=ca.mean, mean_b=cb.mean)
            row["pct"] = (cb.mean - ca.mean) / abs(ca.mean) if ca.mean else None
            if ca.std:
                z = (cb.mean - ca.mean) / ca.std
                row["z"] = z
                row["score"] = abs(z)
                if abs(z) >= 0.5:
                    row["flag"] = "SHIFT"
        if abs(nb - na) >= 0.05:
            row["flag"] = (row["flag"] + " NULLS").strip()
            row["score"] = max(row["score"], abs(nb - na) * 10)
        out.append(row)
    for cb in b.columns:
        if cb.name not in am:
            out.append({"column": cb.name, "flag": "ADDED", "score": 9})
    out.sort(key=lambda r: -r["score"])
    return out
