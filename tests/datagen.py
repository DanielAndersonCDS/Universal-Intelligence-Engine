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


def stats_planted(path, n=30000, seed=21):
    """Planted structure: lognormal column, group effect, U-shape, monotone curve, categorical dependence,
    weekly seasonality, and 6 columns driven by 2 latent factors."""
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta
    grp = rng.choice(["A", "B", "C"], n)
    score = rng.normal(50, 10, n) + np.where(grp == "C", 8, 0)
    region = rng.choice(["east", "west", "north"], n)
    currency = np.where(region == "east", "USD", np.where(region == "west", "EUR", "GBP"))
    flip = rng.random(n) < 0.05
    currency = np.where(flip, rng.choice(["USD", "EUR", "GBP"], n), currency)
    x = rng.uniform(-3, 3, n)
    u = x ** 2 + rng.normal(0, 0.5, n)
    m1 = rng.uniform(0, 10, n)
    mono = np.exp(m1 / 2) + rng.normal(0, 0.2, n)
    skewed = np.exp(rng.normal(3, 1, n))
    f1, f2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    fac = [f1 + rng.normal(0, .3, n), f1 + rng.normal(0, .3, n), f1 + rng.normal(0, .3, n), f2 + rng.normal(0, .3, n), f2 + rng.normal(0, .3, n), f2 + rng.normal(0, .3, n)]
    day = rng.integers(0, 730, n)
    weekday = (day % 7)
    visits = rng.normal(100, 5, n) + np.where(weekday >= 5, 40, 0)
    noise_a, noise_b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "grp", "score", "region", "currency", "x", "u", "m1", "mono", "skewed", "f1", "f2", "f3", "f4", "f5", "f6", "visits", "na", "nb"])
        for i in range(n):
            w.writerow([(date(2023, 1, 2) + timedelta(int(day[i]))).isoformat(), grp[i], round(score[i], 3), region[i], currency[i], round(x[i], 4), round(u[i], 4), round(m1[i], 4),
                        round(mono[i], 4), round(skewed[i], 4)] + [round(c[i], 4) for c in fac] + [round(visits[i], 3), round(noise_a[i], 4), round(noise_b[i], 4)])
    return path


def pure_noise(path, n=20000, seed=33):
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "grp", "cat2", "a", "b", "c", "d", "e"])
        for i in range(n):
            w.writerow([(date(2023, 1, 2) + timedelta(int(rng.integers(0, 730)))).isoformat(), rng.choice(["A", "B", "C"]), rng.choice(["p", "q"]),
                        *[round(float(v), 4) for v in rng.normal(0, 1, 5)]])
    return path


def option_chain(path, spot=100.0, r=0.03, q=0.01, seed=6, bad=True):
    """Option quotes from a Black-Scholes world with a volatility smile; optionally with planted data errors."""
    import math
    from uie.core.quant import derivatives as D
    rng = np.random.default_rng(seed)
    rows = []
    for days in (30, 90, 180):
        T = days / 365
        F = spot * math.exp((r - q) * T)
        for K in range(70, 135, 5):
            m = math.log(K / F)
            iv = 0.20 + 0.30 * m * m - 0.15 * m + 0.01 * math.sqrt(T)
            for kind in ("call", "put"):
                px = D.bs_price(spot, K, T, r, q, iv, kind)
                rows.append([K, kind, round(px, 4), days, 1])
    if bad:
        rows[10][2] += 3.0                       # one stale/inflated quote
        rows[0][2] = 28.0                         # a deep in-the-money call priced below intrinsic value
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strike", "type", "mid", "dte", "n"])
        w.writerows(rows)
    return path
