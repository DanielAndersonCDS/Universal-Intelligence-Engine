"""Chart data for the plot builder (no Qt): each builder returns (spec, rows, sql). The UI only draws."""
from __future__ import annotations
import numpy as np
from . import stats
from .curvefit import fit_models
from .util import qident

KINDS = {
    "hist": "Histogram", "scatter": "Scatter", "line": "Line over time", "bar": "Bar (mean or count by group)", "box": "Box plot by group",
    "heat": "Correlation heatmap", "qq": "Q-Q plot", "acf": "Autocorrelation",
}
NUM = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}


def _num(t):
    t = str(t).upper()
    return t in NUM or t.startswith("DECIMAL")


def _temporal(t):
    t = str(t).upper()
    return t == "DATE" or t.startswith("TIMESTAMP")


def _hash_filter(con, rel, cols, limit):
    n = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    mod = max(1, n // limit)
    return f"AND hash({', '.join(qident(c) for c in cols)}) % {mod} = 0" if mod > 1 else ""


def _unit(span_days: float) -> str:
    return "month" if span_days >= 1100 else "week" if span_days >= 120 else "day"


def build_chart(con, kind: str, x=None, y=None, group=None, bins: int = 40, overlay: bool = False, rel: str = "data", max_points: int = 8000, log_x: bool = False):
    if kind not in KINDS:
        raise ValueError(f"unknown chart kind '{kind}'")
    types = {r[0]: str(r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}

    def need(col, what, pred=None):
        if not col:
            raise ValueError(f"Choose {what}.")
        if col not in types:
            raise ValueError(f"Column '{col}' not found.")
        if pred and not pred(types[col]):
            raise ValueError(f"'{col}' must be {'numeric' if pred is _num else 'a date/time column'} for this chart (it is {types[col]}).")
        return qident(col)

    if kind in ("hist", "qq"):
        qx = need(x, "a numeric X column", _num)
        mn, mx, n = con.execute(f"SELECT min({qx}), max({qx}), count({qx}) FROM {rel}").fetchone()
        if not n or mn == mx:
            raise ValueError(f"'{x}' has no spread to plot.")
        if kind == "hist":
            ex = qx
            if log_x:
                if mn <= 0:
                    raise ValueError(f"'{x}' has values <= 0, so a log scale is not possible.")
                ex = f"ln({qx})"
                mn, mx = float(np.log(mn)), float(np.log(mx))
            w = (mx - mn) / bins
            sql = f"SELECT least(floor(({ex} - ({mn!r})) / {w!r}), {bins - 1})::INTEGER AS bin, count(*) AS rows FROM {rel} WHERE {qx} IS NOT NULL GROUP BY 1 ORDER BY 1"
            spec = {"kind": "hist", "mn": float(mn), "w": float(w), "n": int(n), "xlabel": f"ln({x})" if log_x else x, "ylabel": "rows"}
            if overlay:
                sm = np.array([r[0] for r in con.execute(f"SELECT {qx} FROM {rel} WHERE {qx} IS NOT NULL {_hash_filter(con, rel, [x], 50000)}").fetchall()], float)
                if log_x:
                    lv = np.log(sm[sm > 0])
                    spec["fit"] = {"dist": "normal", "params": {"mu": float(lv.mean()), "sigma": float(lv.std())}}      # normal on ln(x) = lognormal
                else:
                    fits = stats.fit_distributions(sm)
                    if fits:
                        spec["fit"] = {"dist": fits[0]["dist"], "params": fits[0]["params"]}
            return spec, con.execute(sql).fetchall(), sql
        sql = f"SELECT {qx} FROM {rel} WHERE {qx} IS NOT NULL {_hash_filter(con, rel, [x], 20000)}"
        vals = np.array([r[0] for r in con.execute(sql).fetchall()], float)
        dist = "lognormal" if overlay and vals.min() > 0 else "normal"
        pts = stats.qq_points(vals, dist)
        return {"kind": "qq", "xlabel": f"theoretical normal quantiles" + (" (of ln x)" if dist == "lognormal" else ""), "ylabel": f"sample quantiles" + (" (ln x)" if dist == "lognormal" else "")}, pts, sql
    if kind == "scatter":
        qx, qy = need(x, "a numeric X column", _num), need(y, "a numeric Y column", _num)
        sql = f"SELECT {qx} AS x, {qy} AS y FROM {rel} WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL {_hash_filter(con, rel, [x, y], max_points)}"
        rows = con.execute(sql).fetchall()
        if len(rows) < 3:
            raise ValueError("Not enough complete rows.")
        spec = {"kind": "scatter", "xlabel": x, "ylabel": y}
        if overlay:
            arr = np.array(rows, float)
            fits = fit_models(arr[:, 0], arr[:, 1])
            spec["curves"] = [{"name": f["model"], "model": f["model"], "params": f["params"]} for f in fits[:2]]
        return spec, rows, sql
    if kind in ("line", "acf"):
        qx = need(x, "a date/time X column", _temporal)
        mn, mx = con.execute(f"SELECT min({qx}), max({qx}) FROM {rel}").fetchone()
        if mn is None:
            raise ValueError("No dates to plot.")
        unit = _unit((mx - mn).total_seconds() / 86400 if hasattr(mx - mn, "total_seconds") else (mx - mn).days)
        agg = f"avg({need(y, 'a numeric Y column', _num)})" if y else "count(*)"
        sql = f"SELECT CAST(date_trunc('{unit}', {qx}) AS DATE) AS bucket, {agg} AS value FROM {rel} WHERE {qx} IS NOT NULL GROUP BY 1 ORDER BY 1"
        rows = con.execute(sql).fetchall()
        lab = f"{'mean of ' + y if y else 'rows'} per {unit}"
        if kind == "line":
            return {"kind": "line", "xlabel": "time", "ylabel": lab}, rows, sql
        return {"kind": "acf", "maxlag": min(max(len(rows) // 3, 2), 60), "xlabel": f"lag ({unit}s)", "ylabel": "autocorrelation of " + lab}, rows, sql
    if kind == "bar":
        qg = need(group or x, "a group column")
        agg = f"avg({need(y, 'a numeric Y column', _num)})" if y else "count(*)"
        sql = f"SELECT CAST({qg} AS VARCHAR) AS grp, {agg} AS value FROM {rel} WHERE {qg} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
        return {"kind": "bar", "xlabel": group or x, "ylabel": f"mean of {y}" if y else "rows"}, con.execute(sql).fetchall(), sql
    if kind == "box":
        qg, qy = need(group or x, "a group column"), need(y, "a numeric Y column", _num)
        sql = (f"WITH g AS (SELECT CAST({qg} AS VARCHAR) AS grp, {qy} AS v FROM {rel} WHERE {qg} IS NOT NULL AND {qy} IS NOT NULL), "
               f"top AS (SELECT grp FROM g GROUP BY 1 ORDER BY count(*) DESC LIMIT 12), "
               f"s AS (SELECT grp, quantile_cont(v, 0.25) AS q1, quantile_cont(v, 0.5) AS med, quantile_cont(v, 0.75) AS q3, count(*) AS n FROM g WHERE grp IN (SELECT grp FROM top) GROUP BY 1) "
               f"SELECT s.grp, min(v) FILTER (WHERE v >= q1 - 1.5 * (q3 - q1)) AS whisker_lo, q1, med, q3, max(v) FILTER (WHERE v <= q3 + 1.5 * (q3 - q1)) AS whisker_hi, n "
               f"FROM g JOIN s USING (grp) GROUP BY s.grp, q1, med, q3, n ORDER BY med DESC")
        return {"kind": "box", "xlabel": group or x, "ylabel": y}, con.execute(sql).fetchall(), sql
    if kind == "heat":
        cols = [c for c, t in types.items() if _num(t)]
        cols = [c for c in cols if con.execute(f"SELECT stddev_samp({qident(c)}) FROM {rel}").fetchone()[0]][:20]
        if len(cols) < 2:
            raise ValueError("Need at least two numeric columns with variation.")
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
        sql = "SELECT " + ", ".join(f"corr({qident(cols[i])}, {qident(cols[j])})" for i, j in pairs) + f" FROM {rel}"
        row = con.execute(sql).fetchone()
        rows = [(i, i, 1.0) for i in range(len(cols))]
        for (i, j), v in zip(pairs, row):
            v = 0.0 if v is None else float(v)
            rows += [(i, j, v), (j, i, v)]
        return {"kind": "heat", "labels": cols, "xlabel": "", "ylabel": ""}, rows, sql
