"""Principal components of the numeric columns and multicollinearity (variance inflation factors)."""
import numpy as np
from ..facts import Fact
from ..util import qident


def _corr_matrix(ctx, cols):
    k = len(cols)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    row = ctx.con.execute("SELECT " + ", ".join(f"corr({qident(cols[i].name)}, {qident(cols[j].name)})" for i, j in pairs) + f" FROM {ctx.rel}").fetchone()
    R = np.eye(k)
    for (i, j), v in zip(pairs, row):
        R[i, j] = R[j, i] = 0.0 if v is None else float(v)
    return R


def _values_sql(rows):
    return "SELECT * FROM (VALUES " + ", ".join("(" + ", ".join(repr(v) if not isinstance(v, str) else "'" + v.replace("'", "''") + "'" for v in r) + ")" for r in rows) + ") t(x, y)"


def detect(ctx):
    cols = sorted([c for c in ctx.measures("multivariate") if c.distinct > 5], key=lambda c: c.nulls)[:20]
    k = len(cols)
    if k < 4:
        return []
    R = _corr_matrix(ctx, cols)
    names = [c.name for c in cols]
    out = []
    w, V = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0), V[:, order]
    share = w / w.sum()
    k90 = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
    if k90 <= 0.7 * k:
        load = lambda p: [[names[i], float(V[i, p])] for i in np.argsort(-np.abs(V[:, p]))[:3]]
        ev = _values_sql([[f"PC{i + 1}", float(share[i])] for i in range(min(k, 10))])
        out.append(Fact(type="pca", title="Principal components", columns=names,
                        metrics={"columns": k, "components90": k90, "pc1": float(share[0]), "pc2": float(share[1]), "loadings_pc1": load(0), "loadings_pc2": load(1),
                                 "explained": [float(s) for s in share[:6]], "basis": "correlation matrix of the numeric columns (pairwise-complete)"},
                        severity="info", score=0.3 + 0.5 * (1 - k90 / k), evidence_sql=ev,
                        chart={"kind": "bar", "xlabel": "component", "ylabel": "share of variance", "sql": ev}))
    try:
        vif = np.diag(np.linalg.inv(R))
    except np.linalg.LinAlgError:
        vif = np.full(k, np.inf)
    hi = sorted([(float(v), n) for v, n in zip(vif, names) if v >= 10], reverse=True)
    if hi:
        ev = _values_sql([[n, min(v, 1e6)] for v, n in sorted(zip(vif, names), reverse=True)])
        out.append(Fact(type="multicollinearity", title="Collinear columns", columns=[n for _, n in hi],
                        metrics={"count": len(hi), "top": [[n, v if np.isfinite(v) else 1e6] for v, n in hi[:5]], "threshold": 10.0},
                        severity="notable" if hi[0][0] >= 30 else "info", score=min(1.0, np.log10(max(hi[0][0], 10)) / 3), evidence_sql=ev,
                        chart={"kind": "bar", "xlabel": "column", "ylabel": "VIF", "sql": ev}))
    return out
