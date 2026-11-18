"""Curve fitting between two numeric columns: linear, quadratic, cubic, exponential, power, logarithmic.
Models are ranked by AIC on the original scale; transformed models (exponential, power) are evaluated back on the original scale."""
from __future__ import annotations
import math
import numpy as np
from . import provenance
from .config import Config
from .facts import Fact
from .ingest import DataSource
from .semantics import build_clean_view
from .subreport import SubReport
from .util import qident

MODELS = ["linear", "quadratic", "cubic", "exponential", "power", "logarithmic"]


def predict(model: str, p: list, x):
    x = np.asarray(x, float)
    if model in ("linear", "quadratic", "cubic"):
        return np.polyval(p, x)
    with np.errstate(all="ignore"):
        if model == "exponential":
            return p[0] * np.exp(p[1] * x)
        if model == "power":
            return p[0] * np.power(x, p[1])
        if model == "logarithmic":
            return p[0] + p[1] * np.log(x)
    raise ValueError(model)


def equation(model: str, p: list, x: str = "x", y: str = "y") -> str:
    g = lambda v: f"{v:.4g}"
    if model == "linear":
        return f"{y} = {g(p[1])} + {g(p[0])}*{x}"
    if model == "quadratic":
        return f"{y} = {g(p[2])} + {g(p[1])}*{x} + {g(p[0])}*{x}^2"
    if model == "cubic":
        return f"{y} = {g(p[3])} + {g(p[2])}*{x} + {g(p[1])}*{x}^2 + {g(p[0])}*{x}^3"
    if model == "exponential":
        return f"{y} = {g(p[0])} * exp({g(p[1])}*{x})"
    if model == "power":
        return f"{y} = {g(p[0])} * {x}^{g(p[1])}"
    return f"{y} = {g(p[0])} + {g(p[1])}*ln({x})"


def fit_models(x, y) -> list:
    """Fit every applicable model. Returns dicts sorted by AIC (best first)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return []
    sst = float(((y - y.mean()) ** 2).sum())
    out = []

    def add(name, params, k):
        pred = predict(name, params, x)
        if not np.all(np.isfinite(pred)):
            return
        sse = float(((y - pred) ** 2).sum())
        r2 = 1 - sse / sst
        out.append({"model": name, "params": [float(v) for v in params], "k": k, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1),
                    "rmse": math.sqrt(sse / n), "aic": n * math.log(max(sse / n, 1e-300)) + 2 * k})

    xc = (x - x.mean()) / x.std()
    for name, deg in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
        # fit on standardised x for numerical stability, then convert the coefficients back to the original x
        c = np.polyfit(xc, y, deg)
        mu, sd = x.mean(), x.std()
        poly = np.poly1d([1.0])
        conv = np.poly1d([0.0])
        base = np.poly1d([1 / sd, -mu / sd])
        for coef in c:
            conv = conv * base + coef
        add(name, list(conv.coeffs) if len(conv.coeffs) == deg + 1 else [0.0] * (deg + 1 - len(conv.coeffs)) + list(conv.coeffs), deg + 1)
    if np.all(y > 0):
        b, a = np.polyfit(x, np.log(y), 1)
        add("exponential", [math.exp(a), b], 2)
    if np.all(x > 0) and np.all(y > 0):
        b, a = np.polyfit(np.log(x), np.log(y), 1)
        add("power", [math.exp(a), b], 2)
    if np.all(x > 0):
        b, a = np.polyfit(np.log(x), y, 1)
        add("logarithmic", [a, b], 2)
    out.sort(key=lambda d: d["aic"])
    return out


def integrate_trapezoid(x, y, bins: int = 50):
    """Area under the curve of mean(y) over `bins` equal-width x bins, plus the x where the binned mean changes fastest."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum() >= 3:
            xs.append(float(x[sel].mean()))
            ys.append(float(y[sel].mean()))
    if len(xs) < 5:
        return None
    xs, ys = np.array(xs), np.array(ys)
    slope = np.diff(ys) / np.diff(xs)
    k = int(np.argmax(np.abs(slope)))
    return {"area": float(np.trapezoid(ys, xs) if hasattr(np, "trapezoid") else np.trapz(ys, xs)), "max_slope": float(slope[k]),
            "max_slope_at": float((xs[k] + xs[k + 1]) / 2)}


def fit_curves(path, x: str, y: str, cfg: Config | None = None, sem=None, table=None, at=None, sample_rows: int = 50000,
               progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    src = DataSource(view="raw").open(path, table)
    con = src.con
    build_clean_view(con, "raw", "data", sem, qident)
    cols = {r[0]: str(r[1]) for r in con.execute("DESCRIBE data").fetchall()}
    for c in (x, y):
        if c not in cols:
            raise ValueError(f"Column '{c}' not found. Columns: {', '.join(cols)}")
    qx, qy = qident(x), qident(y)
    total = con.execute(f"SELECT count(*) FROM data WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL").fetchone()[0]
    mod = max(1, total // sample_rows)
    where = f"WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL" + (f" AND hash({qx}, {qy}) % {mod} = 0" if mod > 1 else "")
    progress("Fitting", 40)
    rows = con.execute(f"SELECT TRY_CAST({qx} AS DOUBLE), TRY_CAST({qy} AS DOUBLE) FROM data {where}").fetchall()
    X = np.array(rows, dtype=float).reshape(len(rows), 2)
    fits = fit_models(X[:, 0], X[:, 1])
    if not fits:
        raise ValueError("Not enough numeric, non-constant data to fit a curve (need at least 30 complete rows).")
    best, lin = fits[0], next(f for f in fits if f["model"] == "linear")
    rep_fits = sorted(fits, key=lambda f: f["aic"])
    facts = [Fact(type="curve_fit", title=f"Best curve for {y} vs {x}", columns=[x, y], id="C001",
                  metrics={"x": x, "y": y, "model": best["model"], "equation": equation(best["model"], best["params"], x, y), "params": best["params"],
                           "r2": best["r2"], "adj_r2": best["adj_r2"], "aic": best["aic"], "delta_aic": lin["aic"] - best["aic"], "n": int(total),
                           "sampled": mod > 1},
                  severity="notable" if best["model"] != "linear" and lin["aic"] - best["aic"] >= 10 and best["r2"] - lin["r2"] >= 0.05 else "info",
                  score=best["r2"], evidence_sql=f"SELECT corr({qx}, {qy}) AS pearson_r, count(*) AS n FROM data WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL")]
    num = integrate_trapezoid(X[:, 0], X[:, 1])
    summary = {"x": x, "y": y, "rows_used": len(X), "rows_available": int(total), "best_model": best["model"], "best_r2": round(best["r2"], 4),
               "linear_r2": round(lin["r2"], 4), "x_min": float(X[:, 0].min()), "x_max": float(X[:, 0].max())}
    if num:
        summary.update({"area_under_mean_curve": num["area"], "steepest_slope": num["max_slope"], "steepest_slope_at_x": num["max_slope_at"]})
    table_models = [[f["model"], equation(f["model"], f["params"], x, y), f["r2"], f["adj_r2"], f["rmse"], f["aic"], f["aic"] - best["aic"]] for f in rep_fits]
    tables = {"Models": (["model", "equation", "R2", "adj R2", "RMSE", "AIC", "delta AIC"], table_models)}
    warnings = []
    if at:
        lo, hi = float(X[:, 0].min()), float(X[:, 0].max())
        pr = []
        for v in at:
            val = float(predict(best["model"], best["params"], [v])[0])
            pr.append([v, val, "extrapolation" if v < lo or v > hi else "interpolation"])
            if v < lo or v > hi:
                warnings.append(f"x = {v:g} is outside the observed range [{lo:g}, {hi:g}]; extrapolated values are unreliable.")
        tables["Predictions"] = (["x", f"predicted {y} ({best['model']})", "kind"], pr)
    pts = X[np.linspace(0, len(X) - 1, min(len(X), 3000)).astype(int)]
    spec = {"kind": "scatter", "xlabel": x, "ylabel": y, "curves": [{"name": f["model"], "model": f["model"], "params": f["params"]} for f in rep_fits[:3]]}
    r = SubReport(kind="fit", title=f"Curve fit: {y} vs {x}", summary=summary, facts=facts, tables=tables, warnings=warnings, setup_sql=[src.source_sql()],
                  charts={"C001": (spec, [(float(a), float(b)) for a, b in pts])})
    r.provenance = provenance.build([provenance.input_record(src, "data", cfg.hash_data, progress)], {"x": x, "y": y, "sample_rows": sample_rows},
                                    cfg, sem if sem and (sem.columns or sem.dataset) else None, None, facts, command)
    progress("Done", 100)
    return r
