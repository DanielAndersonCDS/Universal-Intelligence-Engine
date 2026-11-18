"""Derivatives pricing: Black-Scholes-Merton with Greeks and implied volatility, Cox-Ross-Rubinstein binomial trees (European and American),
Monte Carlo (European, Asian, barrier), bond analytics (price, yield, duration, convexity), and option-chain diagnostics
(implied volatility, no-arbitrage checks, smile and term structure)."""
from __future__ import annotations
import math
from datetime import date, datetime
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..ingest import DataSource
from ..semantics import build_clean_view
from ..subreport import SubReport
from ..util import qident

N = stats.norm_cdf


def _d12(S, K, T, r, q, sig):
    sd = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / sd
    return d1, d1 - sd


def bs_price(S, K, T, r, q, sig, kind="call"):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d12(S, K, T, r, q, sig)
    if kind == "call":
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def bs_greeks(S, K, T, r, q, sig, kind="call"):
    """Delta, gamma, vega (per 1 vol point), theta (per calendar day), rho (per 1% rate)."""
    d1, d2 = _d12(S, K, T, r, q, sig)
    pdf = stats.norm_pdf(d1)
    eq, er = math.exp(-q * T), math.exp(-r * T)
    sgn = 1 if kind == "call" else -1
    delta = sgn * eq * N(sgn * d1)
    gamma = eq * pdf / (S * sig * math.sqrt(T))
    vega = S * eq * pdf * math.sqrt(T) / 100
    theta = (-S * eq * pdf * sig / (2 * math.sqrt(T)) - sgn * r * K * er * N(sgn * d2) + sgn * q * S * eq * N(sgn * d1)) / 365
    rho = sgn * K * T * er * N(sgn * d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q, kind="call"):
    """Implied volatility by Newton's method with a bisection fallback. NaN if the price violates no-arbitrage bounds."""
    if T <= 0 or price <= 0:
        return float("nan")
    eq, er = math.exp(-q * T), math.exp(-r * T)
    lower = max(S * eq - K * er, 0.0) if kind == "call" else max(K * er - S * eq, 0.0)
    upper = S * eq if kind == "call" else K * er
    if price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")
    sig = max(0.05, min(3.0, math.sqrt(2 * math.pi / T) * price / S))
    for _ in range(50):
        diff = bs_price(S, K, T, r, q, sig, kind) - price
        if abs(diff) < 1e-10:
            return sig
        vega = S * eq * stats.norm_pdf(_d12(S, K, T, r, q, sig)[0]) * math.sqrt(T)
        if vega < 1e-10:
            break
        sig_n = sig - diff / vega
        if not 1e-6 < sig_n < 10:
            break
        sig = sig_n
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, q, mid, kind) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binomial(S, K, T, r, q, sig, kind="call", american=False, steps=500):
    """CRR tree. Returns price, delta, gamma, theta (per day) from the first nodes of the tree."""
    dt = T / steps
    u = math.exp(sig * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d ** j
    V = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    keep = {}
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            Si = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            V = np.maximum(V, (Si - K) if kind == "call" else (K - Si))
        if i in (2, 1):
            keep[i] = V.copy()
    out = {"price": float(V[0])}
    if steps >= 3:
        v1, v2 = keep[1], keep[2]
        out["delta"] = float((v1[0] - v1[1]) / (S * u - S * d))
        h = 0.5 * (S * u * u - S * d * d)
        out["gamma"] = float(((v2[0] - v2[1]) / (S * u * u - S) - (v2[1] - v2[2]) / (S - S * d * d)) / h)
        out["theta"] = float((v2[1] - V[0]) / (2 * dt) / 365)
    return out


def monte_carlo(S, K, T, r, q, sig, kind="call", style="european", n=200000, steps=252, barrier=None, seed=11):
    """GBM Monte Carlo with antithetic variates. style: european | asian (arithmetic average) | down_out | up_out. Returns price and standard error."""
    rng = np.random.default_rng(seed)
    half = n // 2
    steps = 1 if style == "european" else steps
    dt = T / steps
    Z = rng.standard_normal((half, steps))
    Z = np.vstack([Z, -Z])
    logp = np.cumsum((r - q - 0.5 * sig * sig) * dt + sig * math.sqrt(dt) * Z, axis=1)
    paths = S * np.exp(logp)
    ST = paths[:, -1]
    if style == "asian":
        ref = paths.mean(axis=1)
        pay = np.maximum(ref - K, 0) if kind == "call" else np.maximum(K - ref, 0)
    else:
        pay = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
        if style == "down_out":
            pay = pay * (paths.min(axis=1) > barrier)
        elif style == "up_out":
            pay = pay * (paths.max(axis=1) < barrier)
    # antithetic pairs are averaged before taking the standard error
    pair = 0.5 * (pay[:half] + pay[half:]) * math.exp(-r * T)
    return float(pair.mean()), float(pair.std(ddof=1) / math.sqrt(half))


def bond(face, coupon_rate, years, freq, ytm=None, price=None):
    t = np.arange(1, int(round(years * freq)) + 1) / freq
    cf = np.full(len(t), face * coupon_rate / freq)
    cf[-1] += face

    def pv(y):
        return float(np.sum(cf / (1 + y / freq) ** (t * freq)))
    if ytm is None:
        if price is None:
            raise ValueError("Give either a yield or a price.")
        lo, hi = -0.99, 5.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if pv(mid) > price:
                lo = mid
            else:
                hi = mid
        ytm = (lo + hi) / 2
    P = pv(ytm)
    w = cf / (1 + ytm / freq) ** (t * freq) / P
    mac = float(np.sum(w * t))
    mod = mac / (1 + ytm / freq)
    conv = float(np.sum(w * t * (t + 1 / freq)) / (1 + ytm / freq) ** 2)
    return {"price": P, "ytm": ytm, "macaulay": mac, "modified": mod, "convexity": conv, "dv01": mod * P * 1e-4,
            "price_up100": pv(ytm + 0.01), "price_dn100": pv(ytm - 0.01), "approx_up100": P * (-mod * 0.01 + 0.5 * conv * 0.0001), "approx_dn100": P * (mod * 0.01 + 0.5 * conv * 0.0001)}


def _report(kind, title, summary, facts, tables, warnings=None, charts=None, params=None, cfg=None):
    for i, f in enumerate(facts, 1):
        f.id = f"D{i:03d}"
    r = SubReport(kind=kind, title=title, summary=summary, facts=facts, tables=tables, warnings=warnings or [], charts=charts or {})
    r.provenance = provenance.build([], params or {}, cfg or Config(), None, None, facts, None)
    return r


def price_option(model: str, S, K, T, r=0.0, q=0.0, sigma=0.2, kind="call", american=False, style="european", barrier=None, steps=500, paths=200000, cfg=None) -> SubReport:
    for nm, v in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        if not v or v <= 0:
            raise ValueError(f"{nm} must be positive.")
    facts, rows = [], []
    msg = lambda m, **k: Fact(type="q_price", title=f"{model} price", metrics={"message": m, **k}, severity="info", score=0.3)
    base = bs_price(S, K, T, r, q, sigma, kind)
    greeks = bs_greeks(S, K, T, r, q, sigma, kind)
    par = bs_price(S, K, T, r, q, sigma, "call") - bs_price(S, K, T, r, q, sigma, "put") - (S * math.exp(-q * T) - K * math.exp(-r * T))
    desc = f"{kind} S={S:g}, K={K:g}, T={T:g}y, r={r:.2%}, q={q:.2%}, sigma={sigma:.1%}"
    if model == "black-scholes":
        if american:
            raise ValueError("Black-Scholes is European only; use the binomial model for American options.")
        rows = [["price", base]] + [[k, v] for k, v in greeks.items()]
        facts.append(msg(f"Black-Scholes-Merton {desc}: price {base:.4f}. Delta {greeks['delta']:.4f}, gamma {greeks['gamma']:.5f}, vega {greeks['vega']:.4f} per vol point, "
                         f"theta {greeks['theta']:.4f} per day, rho {greeks['rho']:.4f} per 1% rate. Put-call parity residual {par:.1e}."))
    elif model == "binomial":
        b = binomial(S, K, T, r, q, sigma, kind, american, steps)
        eu = binomial(S, K, T, r, q, sigma, kind, False, steps)["price"] if american else b["price"]
        rows = [["price", b["price"]], ["delta", b.get("delta")], ["gamma", b.get("gamma")], ["theta (per day)", b.get("theta")], ["Black-Scholes price (European)", base], ["European tree price", eu]]
        prem = b["price"] - eu
        facts.append(msg(f"{'American' if american else 'European'} {desc}, {steps}-step CRR tree: price {b['price']:.4f} (Black-Scholes European {base:.4f}, tree error {eu - base:+.4f})."
                         + (f" Early-exercise premium {prem:.4f} ({prem / eu:.1%} of the European value)." if american else "")))
    elif model == "monte-carlo":
        pr, se = monte_carlo(S, K, T, r, q, sigma, kind, style, paths, max(steps // 2, 50), barrier)
        rows = [["price", pr], ["standard error", se], ["95% CI low", pr - 1.96 * se], ["95% CI high", pr + 1.96 * se]]
        txt = f"Monte Carlo ({style}) {desc}, {paths:,} antithetic paths: price {pr:.4f} +/- {1.96 * se:.4f} (95%)."
        if style == "european":
            z = (pr - base) / se if se else 0.0
            rows.append(["Black-Scholes price", base])
            txt += f" Black-Scholes gives {base:.4f}, {z:+.1f} standard errors away" + (" (consistent)." if abs(z) < 3 else " (inconsistent: check inputs).")
        elif style in ("down_out", "up_out") and barrier:
            txt += f" Barrier {barrier:g} is monitored discretely ({max(steps // 2, 50)} steps), which overstates the knock-out price relative to continuous monitoring."
        facts.append(msg(txt))
    else:
        raise ValueError("model must be black-scholes, binomial or monte-carlo")
    return _report("price", f"Option pricing: {model}", {"model": model, "option": desc, "style": style if model == "monte-carlo" else ("american" if american else "european")}, facts,
                   {"Result": (["measure", "value"], rows)}, params={"model": model, "S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "kind": kind}, cfg=cfg)


def price_bond(face=100.0, coupon=0.05, years=10.0, freq=2, ytm=None, price=None, cfg=None) -> SubReport:
    b = bond(face, coupon, years, int(freq), ytm, price)
    err_up, err_dn = b["approx_up100"] - b["price_up100"], b["approx_dn100"] - b["price_dn100"]
    f = Fact(type="q_price", title="Bond analytics", severity="info", score=0.3, metrics={"message":
             f"A {years:g}-year {coupon:.2%} bond paying {int(freq)}x per year has price {b['price']:.4f} at a yield of {b['ytm']:.3%}. Macaulay duration {b['macaulay']:.3f} years, modified duration "
             f"{b['modified']:.3f}, convexity {b['convexity']:.2f}, DV01 {b['dv01']:.4f}. A 100 bp rise lowers the price to {b['price_up100']:.4f} ({b['price_up100'] / b['price'] - 1:+.2%}); "
             f"the duration-plus-convexity approximation is off by {err_up:+.4f}."})
    rows = [[k, v] for k, v in b.items()]
    return _report("price", "Bond analytics", {"face": face, "coupon": coupon, "years": years, "payments per year": int(freq), "yield": b["ytm"], "price": b["price"]}, [f], {"Result": (["measure", "value"], rows)},
                   params={"face": face, "coupon": coupon, "years": years, "freq": freq, "ytm": ytm, "price": price}, cfg=cfg)


def _kind(v):
    s = str(v).strip().lower()
    return "call" if s in ("c", "call", "calls", "ca") else "put" if s in ("p", "put", "puts", "pu") else None


def option_chain(path, strike: str, kind: str, price: str = None, bid: str = None, ask: str = None, expiry: str = None, years: str = None, spot: float | str = None,
                 rate: float = 0.0, div: float = 0.0, valuation: str | None = None, table=None, sem=None, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    src = DataSource(view="raw").open(path, table)
    build_clean_view(src.con, "raw", "data", sem, qident)
    cols = {r[0]: str(r[1]) for r in src.con.execute("DESCRIBE data").fetchall()}
    need = [strike, kind] + ([price] if price else [bid, ask]) + [expiry or years] + ([spot] if isinstance(spot, str) else [])
    for c in need:
        if not c or c not in cols:
            raise ValueError(f"Column '{c}' not found. Columns: {', '.join(cols)}")
    if not price and not (bid and ask):
        raise ValueError("Give a price column, or bid and ask columns.")
    px = qident(price) if price else f"({qident(bid)} + {qident(ask)}) / 2.0"
    tex = f"CAST({qident(years)} AS DOUBLE)" if years else (f"date_diff('day', CAST({qident(expiry)} AS DATE), CAST({qident(expiry)} AS DATE))" if False else f"CAST({qident(expiry)} AS DATE)")
    sql = (f"SELECT CAST({qident(strike)} AS DOUBLE) AS strike, CAST({qident(kind)} AS VARCHAR) AS kind, CAST({px} AS DOUBLE) AS price, {tex} AS expiry"
           + (f", CAST({qident(spot)} AS DOUBLE) AS spot" if isinstance(spot, str) else "") + (f", CAST({qident(bid)} AS DOUBLE) AS bid, CAST({qident(ask)} AS DOUBLE) AS ask" if (bid and ask) else "") + " FROM data")
    rows = src.con.execute(sql).fetchall()
    if not rows:
        raise ValueError("No option quotes.")
    if spot is None:
        raise ValueError("Give the underlying price (--spot 123.4 or a column name).")
    val = None
    if not years:
        if valuation:
            val = datetime.strptime(valuation, "%Y-%m-%d").date()
        else:
            raise ValueError("Give the valuation date (--valuation YYYY-MM-DD) when expiries are dates.")
    quotes = []
    bad_kind = 0
    for rw in rows:
        K, kd, pr, ex = rw[0], _kind(rw[1]), rw[2], rw[3]
        S = rw[4] if isinstance(spot, str) else float(spot)
        if kd is None:
            bad_kind += 1
            continue
        T = float(ex) if years else (ex - val).days / 365.0
        if K is None or pr is None or ex is None or S is None or T <= 0:
            continue
        crossed = bool(bid and ask and rw[-2] is not None and rw[-1] is not None and rw[-2] > rw[-1])
        eq_, er_ = math.exp(-div * T), math.exp(-rate * T)
        lower = max(S * eq_ - K * er_, 0.0) if kd == "call" else max(K * er_ - S * eq_, 0.0)
        upper = S * eq_ if kd == "call" else K * er_
        tol_b = max(0.005, 1e-4 * S)                      # price ticks: quotes this close to a bound carry no volatility information, and are not violations
        state = "below" if pr < lower - tol_b else "above" if pr > upper + tol_b else ("flat" if pr <= lower + tol_b else "ok")
        iv = implied_vol(pr, S, K, T, rate, div, kd) if state == "ok" else float("nan")
        quotes.append({"K": K, "kind": kd, "price": pr, "T": T, "S": S, "crossed": crossed, "iv": iv, "state": state})
    if len(quotes) < 6:
        raise ValueError(f"Only {len(quotes)} usable quotes (positive time to expiry, valid type and price).")
    notes = ([f"{bad_kind} rows with an unrecognised option type were skipped."] if bad_kind else [])
    facts = []

    def add(type_, title, msg, sev, score, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=[strike, kind]))

    below = [q for q in quotes if q["state"] in ("below", "above")]
    S0 = float(np.median([q["S"] for q in quotes]))
    tol = max(0.01, 0.002 * S0)
    viol, par_v, mono_v, bfly_v = [], [], [], []
    groups = {}
    for q in quotes:
        groups.setdefault((round(q["T"], 4), q["kind"]), []).append(q)
    for (T, kd), g in groups.items():
        g = sorted(g, key=lambda x: x["K"])
        for a, b in zip(g, g[1:]):
            if b["K"] > a["K"]:
                if (kd == "call" and b["price"] > a["price"] + tol) or (kd == "put" and b["price"] < a["price"] - tol):
                    mono_v.append((T, kd, a["K"], b["K"], a["price"], b["price"]))
        for a, b, c in zip(g, g[1:], g[2:]):
            if a["K"] < b["K"] < c["K"]:
                lim = ((c["K"] - b["K"]) * a["price"] + (b["K"] - a["K"]) * c["price"]) / (c["K"] - a["K"])
                if b["price"] > lim + tol:
                    bfly_v.append((T, kd, a["K"], b["K"], c["K"], b["price"] - lim))
    byk = {}
    for q in quotes:
        byk.setdefault((round(q["T"], 4), q["K"]), {})[q["kind"]] = q
    for (T, K), d in byk.items():
        if "call" in d and "put" in d:
            S = d["call"]["S"]
            resid = d["call"]["price"] - d["put"]["price"] - (S * math.exp(-div * T) - K * math.exp(-rate * T))
            if abs(resid) > tol * 2:
                par_v.append((T, K, resid))
    n = len(quotes)
    if below:
        add("q_opt_bounds", "Prices outside no-arbitrage bounds", f"{len(below)} of {n} quotes ({len(below) / n:.1%}) are below intrinsic value or above the upper bound, so no implied volatility exists "
            f"(for example strike {below[0]['K']:g} {below[0]['kind']} at {below[0]['price']:g}). Check stale prices, the rate/dividend inputs, or the underlying price.", "warning" if len(below) / n > 0.02 else "notable", min(1.0, len(below) / n * 10), count=len(below))
    if mono_v:
        t0 = mono_v[0]
        add("q_opt_arbitrage", "Monotonicity violations", f"{len(mono_v)} adjacent strike pairs break monotonicity (calls must fall and puts must rise with strike), for example the {t0[1]} {t0[2]:g} -> {t0[3]:g} strikes at "
            f"T = {t0[0]:.2f}y priced {t0[4]:g} then {t0[5]:g}. These are arbitrage opportunities on paper and usually stale or crossed quotes.", "warning", 0.8, count=len(mono_v))
    if bfly_v:
        t0 = max(bfly_v, key=lambda v: v[5])
        add("q_opt_butterfly", "Butterfly (convexity) violations", f"{len(bfly_v)} strike triples violate convexity of price in strike (a negative-cost butterfly); the largest is {t0[5]:.3f} for the {t0[1]} "
            f"{t0[2]:g}/{t0[3]:g}/{t0[4]:g} at T = {t0[0]:.2f}y.", "warning", 0.7, count=len(bfly_v))
    if par_v:
        w = max(par_v, key=lambda v: abs(v[2]))
        add("q_opt_parity", "Put-call parity violations", f"{len(par_v)} of {len(byk)} strike/expiry pairs break put-call parity by more than {2 * tol:.2f}; the largest is {w[2]:+.2f} at strike {w[1]:g}, T = {w[0]:.2f}y. "
            f"Persistent one-sided residuals usually mean the rate or dividend inputs are wrong (or the options are American).", "notable", 0.6, count=len(par_v))
    if sum(q["crossed"] for q in quotes):
        add("q_opt_crossed", "Crossed quotes", f"{sum(q['crossed'] for q in quotes)} quotes have bid above ask.", "notable", 0.4)
    smile_rows, term = [], []
    for T in sorted({round(q["T"], 4) for q in quotes}):
        qs = [q for q in quotes if round(q["T"], 4) == T and np.isfinite(q["iv"])]
        if len(qs) < 5:
            continue
        F = qs[0]["S"] * math.exp((rate - div) * T)
        qs.sort(key=lambda x: x["K"])
        ks, ivs = np.array([q["K"] for q in qs]), np.array([q["iv"] for q in qs])
        mon = ks / F
        atm = float(np.interp(1.0, mon, ivs)) if mon.min() <= 1.0 <= mon.max() else float("nan")
        lo_, hi_ = (float(np.interp(0.9, mon, ivs)), float(np.interp(1.1, mon, ivs))) if mon.min() <= 0.9 and mon.max() >= 1.1 else (float("nan"), float("nan"))
        smile_rows.append([T, len(qs), atm, lo_, hi_, lo_ - hi_ if np.isfinite(lo_) and np.isfinite(hi_) else float("nan")])
        if np.isfinite(atm):
            term.append((T, atm))
    sk = [r for r in smile_rows if np.isfinite(r[5])]
    if sk:
        r0 = sk[0]
        add("q_opt_smile", "Volatility skew", f"At T = {r0[0]:.2f}y implied volatility is {r0[3]:.1%} at 90% moneyness versus {r0[4]:.1%} at 110%: a skew of {r0[5] * 100:+.1f} vol points " +
            ("(downside protection is priced richer than upside, the usual equity pattern)." if r0[5] > 0.01 else "(roughly flat or reversed)."), "info", min(1.0, abs(r0[5]) * 5), skew=r0[5])
    if len(term) >= 2:
        s, l = term[0], term[-1]
        add("q_opt_term", "Term structure", f"At-the-money implied volatility is {s[1]:.1%} at T = {s[0]:.2f}y and {l[1]:.1%} at T = {l[0]:.2f}y: the term structure is "
            f"{'inverted (near-term stress is priced)' if s[1] > l[1] * 1.15 else 'upward sloping' if l[1] > s[1] * 1.05 else 'roughly flat'}.", "notable" if s[1] > l[1] * 1.15 else "info", 0.4)
    ok = sum(np.isfinite(q["iv"]) for q in quotes)
    if not facts:
        add("q_opt_clean", "Chain looks consistent", f"{ok} of {n} quotes have implied volatilities and no monotonicity, convexity or parity violations were found at a tolerance of {tol:.2f}.", "info", 0.2)
    for i, f in enumerate(facts, 1):
        f.id = f"O{i:03d}"
    pts = [(math.log(q["K"] / (q["S"] * math.exp((rate - div) * q["T"]))), q["iv"] * 100) for q in quotes if np.isfinite(q["iv"])]
    charts = {facts[0].id: ({"kind": "scatter", "xlabel": "log-moneyness ln(K/F)", "ylabel": "implied volatility, %"}, pts[:5000])}
    qrows = [[q["K"], q["kind"], round(q["T"], 4), q["price"], q["iv"] if np.isfinite(q["iv"]) else None, {"below": "below intrinsic", "above": "above upper bound", "flat": "at intrinsic (no vol info)"}.get(q["state"], "crossed" if q["crossed"] else "")]
             for q in sorted(quotes, key=lambda x: (x["T"], x["kind"], x["K"]))[:5000]]
    summary = {"quotes": n, "with implied vol": ok, "spot": S0, "rate": rate, "dividend yield": div, "expiries": len({round(q['T'], 4) for q in quotes}), "arbitrage tolerance": tol,
               "valuation date": str(val) if val else "maturity given in years"}
    out = SubReport(kind="options", title="Option chain diagnostics", summary=summary, facts=facts, setup_sql=[src.source_sql()], warnings=notes, charts=charts,
                    tables={"Smile by expiry": (["T (years)", "quotes", "ATM IV", "IV at 90%", "IV at 110%", "skew (90% - 110%)"], smile_rows),
                            "Quotes": (["strike", "type", "T", "price", "implied vol", "flag"], qrows),
                            "Monotonicity violations": (["T", "type", "strike 1", "strike 2", "price 1", "price 2"], [list(v) for v in mono_v[:500]])})
    out.provenance = provenance.build([provenance.input_record(src, "data", cfg.hash_data)], {"rate": rate, "div": div, "spot": spot, "valuation": valuation}, cfg, None, None, facts, command)
    return out
