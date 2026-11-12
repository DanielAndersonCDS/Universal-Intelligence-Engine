"""Synthetic datasets with planted patterns, used by the tests."""
import csv
from datetime import date, timedelta
import numpy as np


def transactions(path, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 1)
    days = rng.integers(0, 700, n)
    # a 40-day outage with no transactions
    days = np.where((days > 300) & (days < 340), days + 45, days)
    vendors = np.array([f"V{i:03d}" for i in range(300)])
    vend = vendors[rng.zipf(1.6, n) % 300]
    region = rng.choice(["East", "West", "North"], n, p=[.5, .3, .2])
    amt = np.round(np.exp(rng.normal(5.5, 1.8, n)), 2)
    r = rng.random(n)
    amt = np.where(r < 0.06, rng.choice([1000, 2000, 5000], n), amt)          # round numbers
    near = (r > 0.06) & (r < 0.09)
    amt = np.where(near, np.round(rng.uniform(9500, 9990, n), 2), amt)        # bunching below 10,000
    rows = list(zip([(start + timedelta(int(d))).isoformat() for d in days], vend, amt, region))
    # near-duplicate payments: same vendor and amount two days apart
    for i in range(0, 600, 12):
        d, v, a, rg = rows[i]
        rows.append(((date.fromisoformat(d) + timedelta(2)).isoformat(), v, a, rg))
    # growth in East in the final month so the variance bridge has a clear driver
    last = max(date.fromisoformat(r[0]) for r in rows)
    m0 = date(last.year, last.month, 1)
    for _ in range(800):
        rows.append(((m0 + timedelta(int(rng.integers(0, 20)))).isoformat(), "V001", round(float(rng.uniform(900, 1500)), 2), "East"))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["txn_date", "vendor", "amount", "region"])
        w.writerows(rows)
    return path


def prices(path, seed=3):
    rng = np.random.default_rng(seed)
    days = [date(2022, 1, 3) + timedelta(i) for i in range(1100)]
    days = [d for d in days if d.weekday() < 5]
    n = len(days)
    out = []
    shared = rng.normal(0, 0.014, n)
    for k, t in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        vol = np.full(n, 0.01)
        if t == "CCC":
            vol[int(n * 0.7):] = 0.035                                     # volatility regime shift
        r = rng.normal(0.0004 * (k + 1), 1, n) * vol + (shared if t in ("AAA", "BBB") else 0)
        p = 100 * np.exp(np.cumsum(r))
        out += [(d.isoformat(), t, round(float(x), 4)) for d, x in zip(days, p)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "close"])
        w.writerows(out)
    return path


def ml(path, n=20000, seed=4):
    rng = np.random.default_rng(seed)
    churn = (rng.random(n) < 0.03).astype(int)
    age = rng.normal(40, 12, n).round(1)
    spend = rng.gamma(2, 50, n).round(2) + churn * 5
    leak = churn * 10 + rng.normal(0, 0.05, n)
    reason = np.where(churn == 1, rng.choice(["price", "service", "moved"], n), "")
    city = np.array([f"city{i}" for i in rng.integers(0, 600, n)])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "age", "spend", "leak_score", "cancel_reason", "city", "country", "churn"])
        for i in range(n):
            w.writerow([i, age[i], spend[i], round(float(leak[i]), 4), reason[i], city[i], "US", churn[i]])
    return path
