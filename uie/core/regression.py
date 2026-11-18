"""Ordinary least squares regression from exact sufficient statistics computed in SQL (centred X'X, X'y, y'y over all complete rows),
so it is exact for any row count and never loads the data into memory. Text predictors become dummy variables (most common level = reference)."""
from __future__ import annotations
import math
import numpy as np
from . import provenance, stats
from .config import Config
from .facts import Fact
from .ingest import DataSource
from .semantics import _lit, build_clean_view
from .subreport import SubReport
from .util import qident

NUM = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}


def _is_num(t):
    t = str(t).upper()
    return t in NUM or t.startswith("DECIMAL")


def regress(path, target: str, features, cfg: Config | None = None, sem=None, table=None, max_levels: int = 10,
            progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    features = [features] if isinstance(features, str) else list(features)
    if not features:
        raise ValueError("Give at least one feature column.")
    src = DataSource(view="raw").open(path, table)
    con = src.con
    build_clean_view(con, "raw", "data", sem, qident)
    types = {r[0]: str(r[1]) for r in con.execute("DESCRIBE data").fetchall()}
    for c in [target] + features:
        if c not in types:
            raise ValueError(f"Column '{c}' not found. Columns: {', '.join(types)}")
    if target in features:
        raise ValueError("The target cannot also be a feature.")
    if not _is_num(types[target]):
        raise ValueError(f"Target '{target}' must be numeric (it is {types[target]}). Logistic regression is not implemented yet.")
    cc = " AND ".join(f"{qident(c)} IS NOT NULL" for c in [target] + features)
    terms = []     # (name, sql expression, source column)
    for f in features:
        q = qident(f)
        if _is_num(types[f]):
            terms.append((f, q, f))
        else:
            lv = con.execute(f"SELECT CAST({q} AS VARCHAR), count(*) FROM data WHERE {cc} GROUP BY 1 ORDER BY 2 DESC, 1").fetchall()
            if len(lv) < 2:
                raise ValueError(f"Feature '{f}' has fewer than 2 levels among complete rows.")
            if len(lv) > max_levels:
                raise ValueError(f"Feature '{f}' has {len(lv)} levels (limit {max_levels}); raise --max-levels or bin it.")
            for v, _n in lv[1:]:
                terms.append((f"{f}={v}", f"CAST(CAST({q} AS VARCHAR) = {_lit(v)} AS DOUBLE)", f))
    k = len(terms)
    if k > 40:
        raise ValueError("At most 40 predictor terms are supported.")
    qy = f"CAST({qident(target)} AS DOUBLE)"
    progress("Computing means", 20)
    row = con.execute(f"SELECT count(*), {', '.join(f'avg({e})' for _, e, _ in terms)}, avg({qy}) FROM data WHERE {cc}").fetchone()
    n, means, ybar = int(row[0]), [float(v) for v in row[1:1 + k]], float(row[-1])
    if n <= k + 2:
        raise ValueError(f"Only {n} complete rows for {k} predictor terms.")
    progress("Computing cross-products", 50)
    cen = [f"({e} - ({m!r}))" for (_, e, _), m in zip(terms, means)]
    cy = f"({qy} - ({ybar!r}))"
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    exprs = [f"sum({cen[i]} * {cen[j]})" for i, j in pairs] + [f"sum({cen[i]} * {cy})" for i in range(k)] + [f"sum({cy} * {cy})"]
    vals = [float(v or 0.0) for v in con.execute(f"SELECT {', '.join(exprs)} FROM data WHERE {cc}").fetchone()]
    xtx = np.zeros((k, k))
    for (i, j), v in zip(pairs, vals):
        xtx[i, j] = xtx[j, i] = v
    xty, yty = np.array(vals[len(pairs):len(pairs) + k]), vals[-1]
    names = [t[0] for t in terms]
    res = stats.ols_from_gram(xtx, xty, yty, n, means, ybar, names)
    if res is None:
        raise ValueError("The design matrix is singular: some predictors are exact combinations of others (or constant). Remove redundant columns.")
    # residual diagnostics against the fitted values (one more pass)
    fitted = f"({res['intercept']!r} + " + " + ".join(f"({float(b)!r}) * {e}" for b, (_, e, _) in zip(res["beta"], terms)) + ")"
    resid = f"({qy} - {fitted})"
    d = con.execute(f"SELECT skewness({resid}), kurtosis({resid}), corr(abs({resid}), {fitted}), min({resid}), max({resid}) FROM data WHERE {cc}").fetchone()
    skew, kurt, het = (float(d[0] or 0), float(d[1] or 0), float(d[2] or 0))
    jb = n / 6 * (skew ** 2 + kurt ** 2 / 4)
    jb_p = stats.chi2_sf(jb, 2)
    facts = [Fact(type="regress_fit", title=f"Model fit for {target}", columns=[target] + features, id="M001",
                  metrics={"target": target, "r2": res["r2"], "adj_r2": res["adj_r2"], "rmse": res["rmse"], "n": n, "f": res["f"], "f_p": res["f_p"], "terms": k},
                  severity="info", score=res["r2"],
                  evidence_sql=f"SELECT count(*) AS n, avg({qy}) AS mean_{target.replace(' ', '_')} FROM data WHERE {cc}")]
    order = sorted(range(k), key=lambda i: -abs(res["std_beta"][i]))
    for i in order[:6]:
        if res["p"][i] < 0.001:
            facts.append(Fact(type="regress_predictor", title=f"Predictor {names[i]}", columns=[terms[i][2], target],
                              metrics={"target": target, "term": names[i], "coef": float(res["beta"][i]), "ci_lo": float(res["ci"][i, 0]), "ci_hi": float(res["ci"][i, 1]),
                                       "std_beta": float(res["std_beta"][i]), "p": float(res["p"][i])},
                              severity="info", score=min(1.0, abs(res["std_beta"][i])), evidence_sql=f"SELECT corr({terms[i][1]}, {qy}) AS pearson_r FROM data WHERE {cc}"))

    def warn(title, msg, sev="notable"):
        facts.append(Fact(type="regress_warning", title=title, metrics={"message": msg}, severity=sev, score=0.5, id=""))

    hi = [(names[i], float(res["vif"][i])) for i in range(k) if res["vif"][i] >= 10]
    if hi:
        warn("Collinear predictors", "Collinear predictors (VIF above 10): " + ", ".join(f"`{n_}` ({v:.0f})" for n_, v in sorted(hi, key=lambda t: -t[1])[:5]) +
             ". Their individual coefficients are unstable even if the model predicts well.")
    if jb_p < 0.001 and (abs(skew) >= 1 or abs(kurt) >= 3):
        warn("Non-normal residuals", f"Residuals are not normally distributed (skewness {skew:+.2f}, excess kurtosis {kurt:.1f}); p-values and confidence intervals "
                                     f"assume they are. Consider transforming `{target}` (for example log) or using robust methods.")
    if abs(het) >= 0.2:
        warn("Heteroscedastic residuals", f"The residual spread changes with the fitted value (correlation of |residual| with the fitted value is {het:+.2f}); "
                                          f"standard errors are likely too small or too large.")
    if res["r2"] > 0.999:
        warn("Near-perfect fit", f"R-squared is {res['r2']:.4f}. A predictor may be derived from `{target}` (leakage) or the relationship may be an identity.", "warning")
    for i, f in enumerate(facts):
        f.id = f.id or f"M{i + 1:03d}"
    coef_rows = [["(intercept)", res["intercept"], None, None, None, None, None, None, None]] + [
        [names[i], float(res["beta"][i]), float(res["se"][i]), float(res["t"][i]), float(res["p"][i]), float(res["ci"][i, 0]), float(res["ci"][i, 1]),
         float(res["std_beta"][i]), float(res["vif"][i])] for i in range(k)]
    sample_sql = (f"SELECT {fitted} AS x, {qy} AS y FROM data WHERE {cc} " + (f"AND hash({', '.join(e for _, e, _ in terms[:3])}) % {max(1, n // 4000)} = 0" if n > 4000 else ""))
    pts = con.execute(sample_sql).fetchall()
    summary = {"target": target, "features": ", ".join(features), "rows_used": n, "terms": k, "R2": res["r2"], "adj_R2": res["adj_r2"], "RMSE": res["rmse"],
               "F": res["f"], "F p-value": res["f_p"], "residual skewness": skew, "residual excess kurtosis": kurt}
    r = SubReport(kind="regress", title=f"OLS regression: {target}", summary=summary, facts=facts, setup_sql=[src.source_sql()],
                  tables={"Coefficients": (["term", "coef", "std err", "t", "p", "CI 2.5%", "CI 97.5%", "std beta", "VIF"], coef_rows)},
                  charts={"M001": ({"kind": "scatter", "xlabel": f"fitted {target}", "ylabel": f"actual {target}", "curves": [{"name": "y = x", "model": "linear", "params": [1.0, 0.0]}]},
                                   [(float(a), float(b)) for a, b in pts[:4000]])})
    r.provenance = provenance.build([provenance.input_record(src, "data", cfg.hash_data, progress)], {"target": target, "features": features, "max_levels": max_levels},
                                    cfg, sem if sem and (sem.columns or sem.dataset) else None, None, facts, command)
    if n < 10 * k:
        r.warnings.append(f"Only {n} rows for {k} terms; estimates will be noisy.")
    progress("Done", 100)
    return r
