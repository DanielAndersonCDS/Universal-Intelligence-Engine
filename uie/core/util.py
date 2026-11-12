from __future__ import annotations
from datetime import datetime, date, timezone


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def human_count(n: float) -> str:
    n = float(n)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.2f}{suf}"
    return f"{n:,.0f}"


def fmt_num(x) -> str:
    if x is None:
        return "n/a"
    x = float(x)
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_ts(v) -> float:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.timestamp()
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, tzinfo=timezone.utc).timestamp()
    return float(v)


def ts_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
