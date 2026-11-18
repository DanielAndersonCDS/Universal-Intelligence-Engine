"""Association that Pearson's r misses (monotonic-but-curved, U-shaped) and association between categorical columns (Cramer's V)."""
import itertools
import numpy as np
from ..facts import Fact
from ..util import qident
from .. import stats


def detect(ctx):
    return _numeric(ctx) + _categorical(ctx)


def _r2_quad(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50 or x.std() == 0 or y.std() == 0:
        return 0.0
    xs = (x - x.mean()) / x.std()
    A = np.column_stack([np.ones(len(xs)), xs, xs ** 2])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - A @ coef) ** 2).sum()) / sst if sst else 0.0


def _numeric(ctx):
    cols = sorted([c for c in ctx.measures("association") if c.distinct > 10], key=lambda c: c.nulls)[:14]
    if len(cols) < 2:
        return []
    X = np.array(ctx.sample_rows([c.name for c in cols], 30000), dtype=float)
    X = X.reshape(len(X), len(cols))
    found = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        a, b = cols[i], cols[j]
        if ctx.is_derived_pair(a.name, b.name):
            continue
        x, y = X[:, i], X[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100 or x[m].std() == 0 or y[m].std() == 0:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        if abs(r) >= 0.7:
            continue                                    # already reported by the correlation detector
        rho = stats.spearman(x, y)
        q = max(_r2_quad(x, y), _r2_quad(y, x))
        kind = None
        if rho is not None and abs(rho) >= 0.7 and abs(rho) - abs(r) >= 0.15:
            kind = "monotonic"
        elif q >= 0.3 and q - r * r >= 0.25:
            kind = "curved"
        if kind:
            found.append((max(abs(rho or 0) - abs(r), q - r * r), a, b, r, rho, q, int(m.sum()), kind))
    found.sort(key=lambda t: -t[0])
    out = []
    for gain, a, b, r, rho, q, n, kind in found[:6]:
        qa, qb, rel = qident(a.name), qident(b.name), ctx.rel
        out.append(Fact(
            type="nonlinear_association", title=f"{a.name} vs {b.name} (nonlinear)", columns=[a.name, b.name],
            metrics={"a": a.name, "b": b.name, "pearson": r, "spearman": rho, "r2_linear": r * r, "r2_quadratic": q, "n": n, "kind": kind},
            severity="notable" if gain >= 0.4 else "info", score=min(1.0, gain),
            evidence_sql=f"SELECT corr({qa}, {qb}) AS pearson_r, count(*) AS n FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL",
            chart={"kind": "scatter", "xlabel": a.name, "ylabel": b.name, "curve": "quadratic",
                   "sql": f"SELECT x, y FROM (SELECT {qa} AS x, {qb} AS y FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL) USING SAMPLE 5000 ROWS"}))
    return out


def _cramers_v(table):
    t = np.asarray(table, float)
    n = t.sum()
    r, k = t.shape
    if n < 50 or r < 2 or k < 2:
        return None
    exp = np.outer(t.sum(1), t.sum(0)) / n
    chi2 = float(((t - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return (math_sqrt(phi2c / denom) if denom > 0 else 0.0), chi2, (r - 1) * (k - 1), int(n)


def math_sqrt(x):
    return float(np.sqrt(x))


def _categorical(ctx):
    rows = ctx.profile.rows
    cats = sorted([c for c in ctx.profile.columns if c.kind == "text" and 2 <= c.distinct <= 30 and not ctx.is_id_like(c)
                   and ctx.sem.role(c.name) not in ("ignore", "id") and c.nulls < 0.3 * rows], key=lambda c: c.nulls)[:8]
    out = []
    for a, b in itertools.combinations(cats, 2):
        qa, qb, rel = qident(a.name), qident(b.name), ctx.rel
        rs = ctx.con.execute(f"SELECT CAST({qa} AS VARCHAR), CAST({qb} AS VARCHAR), count(*) FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL GROUP BY 1, 2").fetchall()
        ra, rb = sorted({r[0] for r in rs}), sorted({r[1] for r in rs})
        t = np.zeros((len(ra), len(rb)))
        ia, ib = {v: i for i, v in enumerate(ra)}, {v: i for i, v in enumerate(rb)}
        for x, y, n in rs:
            t[ia[x], ib[y]] = n
        res = _cramers_v(t)
        if not res:
            continue
        v, chi2, df, n = res
        p = stats.chi2_sf(chi2, df)
        if v >= 0.3 and p < 0.001:
            out.append(Fact(type="categorical_association", title=f"{a.name} vs {b.name} (categorical)", columns=[a.name, b.name],
                            metrics={"a": a.name, "b": b.name, "cramers_v": v, "chi2": chi2, "df": df, "p": p, "n": n},
                            severity="notable" if v >= 0.6 else "info", score=min(1.0, v),
                            evidence_sql=f"SELECT {qa} AS a, {qb} AS b, count(*) AS rows FROM {rel} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL GROUP BY 1, 2 ORDER BY 3 DESC"))
    out.sort(key=lambda f: -f.score)
    return out[:5]
