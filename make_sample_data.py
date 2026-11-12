"""Generates sample_sales.csv with planted trend, level shift, outliers, correlations, concentration, nulls, duplicates."""
import csv
import numpy as np

rng = np.random.default_rng(7)
import sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
days = rng.integers(0, 1095, n)
secs = rng.integers(0, 86400, n)
ts = (np.datetime64("2023-01-01") + days.astype("timedelta64[D]")).astype("datetime64[s]") + secs.astype("timedelta64[s]")
region = rng.choice(["East", "West", "North", "South"], n, p=[.4, .25, .2, .15])
mult = np.where(region == "East", 2.5, 1.0)
segment = rng.choice(["Retail", "Wholesale", "Online"], n, p=[.5, .2, .3])
units = rng.poisson(5, n) + 1
price = rng.normal(40, 8, n).clip(5)
revenue = units * price * (1 + 0.0006 * days) * np.where(days >= 700, 1.35, 1.0) * mult
spike = rng.random(n) < 0.003
revenue = np.where(spike, revenue * 40, revenue)
cost = revenue * 0.6 + rng.normal(0, 3, n)
discount = (0.35 - price / 200 + rng.normal(0, 0.02, n)).clip(0, 0.5)
age = rng.normal(41, 12, n).clip(18, 90).astype(int)
rows = []
with open("sample_sales.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["order_id", "order_date", "region", "segment", "units", "unit_price", "revenue", "cost", "discount", "customer_age"])
    dates = np.datetime_as_string(ts, unit="s")
    for i in range(n):
        row = [i + 1, dates[i].replace("T", " "), region[i], segment[i], units[i], round(price[i], 2),
               round(revenue[i], 2), round(cost[i], 2), round(discount[i], 3), "" if rng.random() < 0.04 else age[i]]
        w.writerow(row)
        if i < 1500:
            rows.append(row)
    w.writerows(rows)
print("wrote sample_sales.csv")
