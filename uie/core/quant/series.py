"""Load one price/return series (optionally one entity of a long table) or a panel of series from a dataset via DuckDB."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .. import provenance
from ..config import Config
from ..ingest import DataSource
from ..semantics import _lit, build_clean_view
from ..util import qident

NUM = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}


def _is_num(t):
    t = str(t).upper()
    return t in NUM or t.startswith("DECIMAL")


def periods_per_year(dates) -> float:
    if len(dates) < 3:
        return 252.0
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    raw = (len(dates) - 1) / years
    for snap in (252, 365, 52, 12, 4, 1):
        if abs(raw - snap) / snap < 0.08:
            return float(snap)
    return float(raw)


@dataclass
class Series:
    name: str
    dates: list
    values: np.ndarray
    kind: str                      # 'price' or 'returns'
    ppy: float
    sql: str
    source: DataSource
    notes: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.values)

    def simple_returns(self):
        """Simple returns aligned to dates[1:] (or all dates when the series already holds returns)."""
        if self.kind == "returns":
            return self.dates, self.values
        v = self.values
        return self.dates[1:], v[1:] / v[:-1] - 1.0

    def log_returns(self):
        if self.kind == "returns":
            return self.dates, np.log1p(self.values)
        return self.dates[1:], np.diff(np.log(self.values))


def _open(path, table, sem):
    src = DataSource(view="raw").open(path, table)
    build_clean_view(src.con, "raw", "data", sem, qident)
    return src


def resolve_columns(src_cols: dict, sem, time=None, value=None, entity=None):
    time = time or (sem.get("time") if sem else None)
    if not time:
        tc = [n for n, t in src_cols.items() if str(t).upper() == "DATE" or str(t).upper().startswith("TIMESTAMP")]
        time = tc[0] if tc else None
    if not value and sem:
        pr = [n for n in sem.cols_with_role("price") if n in src_cols]
        value = pr[0] if pr else None
    if not value:
        for n in src_cols:
            if n.lower() in ("close", "adj_close", "adjclose", "price", "px_last", "nav", "last"):
                value = n
                break
    entity = entity or (sem.get("entity") if sem else None)
    return time, value, entity


def load_series(path, time=None, value=None, entity=None, entity_value=None, table=None, sem=None, unit: str = "day",
                value_is: str = "price", min_obs: int = 60, cfg: Config | None = None, progress=lambda s, p=0: None) -> Series:
    src = _open(path, table, sem)
    cols = {r[0]: str(r[1]) for r in src.con.execute("DESCRIBE data").fetchall()}
    time, value, entity = resolve_columns(cols, sem, time, value, entity)
    for what, c in (("time", time), ("value", value)):
        if not c:
            raise ValueError(f"No {what} column found; pass --{'time' if what == 'time' else 'value'} (columns: {', '.join(cols)}).")
        if c not in cols:
            raise ValueError(f"Column '{c}' not found. Columns: {', '.join(cols)}")
    if not _is_num(cols[value]):
        raise ValueError(f"Value column '{value}' must be numeric (it is {cols[value]}).")
    qt, qv = qident(time), qident(value)
    flt = ""
    label = value
    if entity:
        if entity not in cols:
            raise ValueError(f"Entity column '{entity}' not found.")
        if entity_value is None:
            top = [r[0] for r in src.con.execute(f"SELECT {qident(entity)} FROM data GROUP BY 1 ORDER BY count(*) DESC LIMIT 8").fetchall()]
            raise ValueError(f"'{entity}' has several values; choose one with --entity-value (largest: {', '.join(map(str, top))}).")
        ev = entity_value
        if _is_num(cols[entity]):
            try:
                ev = float(entity_value)
                ev = int(ev) if ev == int(ev) else ev
            except (TypeError, ValueError):
                pass
        flt = f"AND {qident(entity)} = {_lit(ev)}"
        label = f"{value} ({entity}={entity_value})"
    bucket = f"CAST(date_trunc('{unit}', {qt}) AS DATE)" if unit in ("day", "week", "month") else f"CAST({qt} AS DATE)"
    if not entity:       # several rows sharing one timestamp mean the table holds several series (long format): never mix them silently
        n_rows, n_times = src.con.execute(f"SELECT count(*), count(DISTINCT {qt}) FROM data WHERE {qv} IS NOT NULL AND {qt} IS NOT NULL").fetchone()
        if n_rows > 1.02 * n_times:
            cands = [n for n, t in cols.items() if str(t).upper() == "VARCHAR" and n not in (time, value)][:5]
            raise ValueError(f"{n_rows - n_times:,} rows repeat a timestamp, so '{value}' looks like several series stacked in one table. Choose the entity column with --entity "
                             f"and the series with --entity-value" + (f" (candidates: {', '.join(cands)})." if cands else "."))
    sql = f"SELECT {bucket} AS bucket, arg_max({qv}, {qt}) AS value FROM data WHERE {qv} IS NOT NULL AND {qt} IS NOT NULL {flt} GROUP BY 1 ORDER BY 1"
    progress("Loading series", 20)
    rows = src.con.execute(sql).fetchall()
    if len(rows) < min_obs:
        raise ValueError(f"Only {len(rows)} observations; at least {min_obs} are needed.")
    dates = [r[0] for r in rows]
    vals = np.array([float(r[1]) for r in rows])
    notes = []
    if value_is == "price" and np.any(vals <= 0):
        raise ValueError(f"'{value}' has values <= 0, so it cannot be a price series (use --returns if it holds returns).")
    if value_is == "returns" and np.nanmax(np.abs(vals)) > 1.0:
        notes.append(f"'{value}' has values above 100% in magnitude; check that it holds returns as decimals (0.01 = 1%), not percents.")
    return Series(label, dates, vals, value_is, periods_per_year(dates), sql, src, notes)


@dataclass
class Panel:
    dates: list
    names: list
    X: np.ndarray                  # T x N values (prices or returns)
    kind: str
    ppy: float
    sql: str
    source: DataSource
    notes: list = field(default_factory=list)

    def simple_returns(self):
        if self.kind == "returns":
            return self.dates, self.X
        return self.dates[1:], self.X[1:] / self.X[:-1] - 1.0


def load_panel(path, time=None, value=None, entity=None, columns=None, table=None, sem=None, unit: str = "day", value_is: str = "price",
               max_assets: int = 30, min_obs: int = 60, progress=lambda s, p=0: None) -> Panel:
    """Long format (entity + value columns) or wide format (`columns` = one column per asset). Rows are aligned on common dates."""
    src = _open(path, table, sem)
    cols = {r[0]: str(r[1]) for r in src.con.execute("DESCRIBE data").fetchall()}
    time, value, entity = resolve_columns(cols, sem, time, value, None if columns else entity)
    if not time or time not in cols:
        raise ValueError("A time column is required (--time).")
    qt = qident(time)
    bucket = f"CAST(date_trunc('{unit}', {qt}) AS DATE)" if unit in ("day", "week", "month") else f"CAST({qt} AS DATE)"
    notes = []
    if columns:
        for c in columns:
            if c not in cols or not _is_num(cols[c]):
                raise ValueError(f"Asset column '{c}' must exist and be numeric.")
        names = list(columns)
        sql = f"SELECT {bucket} AS bucket, " + ", ".join(f"arg_max({qident(c)}, {qt}) AS {qident(c)}" for c in names) + f" FROM data WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1"
    else:
        if not entity or entity not in cols or not value or value not in cols:
            raise ValueError("Long format needs --entity and --value columns; wide format needs --columns a,b,c.")
        ents = [r[0] for r in src.con.execute(f"SELECT {qident(entity)} FROM data WHERE {qident(entity)} IS NOT NULL AND {qident(value)} IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {max_assets}").fetchall()]
        if len(ents) < 2:
            raise ValueError(f"'{entity}' needs at least two values to build a portfolio.")
        names = [str(e) for e in ents]
        sql = (f"SELECT {bucket} AS bucket, " + ", ".join(f"arg_max({qident(value)}, {qt}) FILTER (WHERE {qident(entity)} = {_lit(e)}) AS {qident(str(e))}" for e in ents) +
               f" FROM data WHERE {qt} IS NOT NULL GROUP BY 1 ORDER BY 1")
    progress("Loading panel", 25)
    rows = src.con.execute(sql).fetchall()
    arr = np.array([[np.nan if v is None else float(v) for v in r[1:]] for r in rows], float)
    dates = [r[0] for r in rows]
    keep = np.mean(np.isfinite(arr), axis=0) >= 0.8
    if keep.sum() < 2:
        raise ValueError("Fewer than two assets have at least 80% coverage on the common calendar.")
    if not keep.all():
        notes.append("Dropped for sparse coverage (<80% of dates): " + ", ".join(n for n, k in zip(names, keep) if not k))
    arr, names = arr[:, keep], [n for n, k in zip(names, keep) if k]
    ok = np.all(np.isfinite(arr), axis=1)
    if ok.mean() < 0.9:
        notes.append(f"Aligned on common dates: kept {int(ok.sum()):,} of {len(ok):,} dates ({ok.mean():.0%}).")
    arr, dates = arr[ok], [d for d, k in zip(dates, ok) if k]
    if len(dates) < min_obs:
        raise ValueError(f"Only {len(dates)} common dates; at least {min_obs} are needed.")
    if value_is == "price" and np.any(arr <= 0):
        raise ValueError("Price panel contains values <= 0.")
    return Panel(dates, names, arr, value_is, periods_per_year(dates), sql, src, notes)


def input_record(src, cfg=None, progress=lambda s, p=0: None):
    return provenance.input_record(src, "data", (cfg or Config()).hash_data, progress)
