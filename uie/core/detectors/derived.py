"""Derived-column detection: is one column the sum/difference or the product/ratio of others?
Screens on a deterministic row sample with least squares, then verifies the exact relation on all rows in SQL.
Detected relations are stored on the context so the correlation detector can suppress tautological pairs."""
from __future__ import annotations
import numpy as np
from ..facts import Fact
from ..util import qident


def _sample(ctx, cols, limit):
    rows = ctx.profile.rows
    mod = max(1, rows // limit)
    names = [c.name for c in ctx.profile.columns][:12]
    sel = ", ".join(qident(c.name) for c in cols)
    where = f"WHERE hash({', '.join(qident(n) for n in names)}) % {mod} = 0" if mod > 1 else ""
    data = ctx.con.execute(f"SELECT {sel} FROM {ctx.rel} {where}").fetchall()
    return np.array(data, dtype=float).reshape(len(data), len(cols))


def _sum_fit(X, t, cand, max_terms=4):
    y = X[:, t]
    vt = np.isfinite(y)
    if vt.sum() < 200:
        return None
    chosen, score = [], None
    for _ in range(max_terms):
        best = None
        for j in cand:
            if j in chosen or j == t:
                continue
            idx = chosen + [j]
            m = vt & np.all(np.isfinite(X[:, idx]), axis=1)
            if m.sum() < max(200, 0.3 * vt.sum()):
                continue
            A, yy = X[m][:, idx], y[m]
            coef = np.linalg.lstsq(A, yy, rcond=None)[0]
            sst = float(((yy - yy.mean()) ** 2).sum())
            if sst <= 0:
                continue
            sc = float(((yy - A @ coef) ** 2).sum()) / sst
            if best is None or sc < best[0]:
                best = (sc, j)
        if best is None:
            break
        chosen.append(best[1])
        score = best[0]
        if score < 1e-8:
            break
    if not chosen or score is None or score > 1e-4:
        return None
    m = vt & np.all(np.isfinite(X[:, chosen]), axis=1)
    coef = np.linalg.lstsq(X[m][:, chosen], y[m], rcond=None)[0]
    if np.any(np.abs(np.abs(coef) - 1) > 0.02):
        return None
    signs = np.sign(coef)
    resid = np.abs(y[m] - X[m][:, chosen] @ signs)
    if np.quantile(resid, 0.99) > 0.01 * (np.std(y[m]) or 1.0):
        return None
    return chosen, [int(s) for s in signs]


def _prod_fit(X, t, cand, limit=4000):
    y = X[:, t]
    vt = np.isfinite(y) & (y != 0)
    if vt.sum() < 200:
        return None
    sel = np.flatnonzero(vt)[:limit]
    ys = y[sel]
    best = None
    for i, a in enumerate(cand):
        if a == t:
            continue
        xa = X[sel, a]
        for b in cand[i + 1:]:
            if b == t:
                continue
            p = xa * X[sel, b]
            ok = np.isfinite(p)
            if ok.sum() < 200:
                continue
            rel = np.abs(ys[ok] - p[ok]) / np.abs(ys[ok])
            if np.median(rel) <= 1e-3 and np.mean(rel <= 5e-3) >= 0.97:
                sc = float(np.median(rel))
                if best is None or sc < best[0]:
                    best = (sc, [a, b])
    return best[1] if best else None


def _signed(items, signs):
    out = ""
    for i, (it, sg) in enumerate(zip(items, signs)):
        out += (it if sg > 0 else f"-{it}") if i == 0 else (f" + {it}" if sg > 0 else f" - {it}")
    return out


def detect(ctx):
    cfg = ctx.cfg
    cols = [c for c in ctx.profile.columns
            if c.kind == "numeric" and c.std and c.std > 0 and c.distinct > 5 and not ctx.is_id_like(c)
            and ctx.sem.role(c.name) not in ("id", "ignore", "entity", "time") and c.nulls < 0.6 * ctx.profile.rows]
    cols = sorted(cols, key=lambda c: c.nulls)[:cfg.derived_max_cols]
    # user-declared derivations always count
    for name, srcs in ctx.sem.derived_sources().items():
        ctx.derived.setdefault(name, list(srcs))
    if len(cols) < 3 or ctx.profile.rows < 200:
        return []
    X = _sample(ctx, cols, cfg.derived_sample_rows)
    idx = list(range(len(cols)))
    mean_abs = np.nanmean(np.abs(X), axis=0)
    found = {}   # frozenset(columns) -> (target, sources, signs, op)
    for t in idx:
        r = _sum_fit(X, t, idx)
        if r:
            chosen, signs = r
            key = frozenset([t] + chosen)
            cur = found.get(key)
            if cur is None or mean_abs[t] > mean_abs[cur[0]]:
                found[key] = (t, chosen, signs, "sum")
    for t in idx:
        r = _prod_fit(X, t, idx)
        if r:
            key = frozenset([t] + r)
            if key not in found:
                found[key] = (t, r, [1, 1], "product")
    out = []
    for key, (t, chosen, signs, op) in found.items():
        if op == "sum":      # re-orient so the largest column (usually the total) is the one described as derived
            coef = {t: -1, **dict(zip(chosen, signs))}
            k = max(coef, key=lambda j: mean_abs[j])
            if k != t:
                chosen = [j for j in coef if j != k]
                signs = [int(-coef[j] / coef[k]) for j in chosen]
                t = k
        tn, sn = cols[t].name, [cols[j].name for j in chosen]
        qt, qs = qident(tn), [qident(n) for n in sn]
        if op == "sum":
            expr_sql, expr_txt = _signed(qs, signs), f"{tn} = {_signed(sn, signs)}"
            std_t = cols[t].std or 1.0
            tol = min(0.0051 * (len(sn) + 1), 0.01 * std_t) + 1e-9
            cond = f"abs({qt} - ({expr_sql})) <= {tol!r}"
        else:
            expr_sql = " * ".join(qs)
            expr_txt = f"{tn} = " + " x ".join(sn)
            cond = f"abs({qt} - ({expr_sql})) <= 0.005 * abs({qt}) + 1e-9"
        notnull = " AND ".join(f"{q} IS NOT NULL" for q in [qt] + qs)
        row = ctx.con.execute(f"SELECT count_if({cond}), count(*) FROM {ctx.rel} WHERE {notnull}").fetchone()
        ok, n = int(row[0] or 0), int(row[1] or 0)
        if n < 100 or ok / n < (0.99 if op == "sum" else 0.97):
            continue
        ctx.derived.setdefault(tn, [])
        ctx.derived[tn] = sorted(set(ctx.derived[tn]) | set(sn))
        out.append(Fact(type="quality_derived", title=f"Derived column {tn}",
                        metrics={"column": tn, "sources": sn, "op": op, "expr": expr_txt, "match_share": ok / n, "n": n},
                        severity="info", score=0.4, columns=[tn] + sn,
                        evidence_sql=f"SELECT count_if({cond}) AS matching, count(*) AS rows_checked FROM {ctx.rel} WHERE {notnull}"))
    return out
