"""GARCH(1,1) and GJR-GARCH(1,1) with normal or Student-t innovations by maximum likelihood (returns in percent).
Stationarity (alpha + beta + gamma/2 < 1) is enforced by a softmax parametrisation. Reports persistence, half-life, long-run and current
volatility, leverage and tail evidence, residual diagnostics, and an out-of-sample volatility forecast comparison against EWMA."""
from __future__ import annotations
import math
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from . import optim
from .series import Series


def _linrec(a, beta, s0):
    """s[t] = a[t] + beta * s[t-1], s[0] = s0, vectorised in blocks (all terms are positive so there is no cancellation)."""
    n = len(a)
    out = np.empty(n)
    out[0] = s0
    beta = min(max(beta, 1e-6), 0.999999)
    B = max(10, int(min(n, 25 / max(-math.log(beta), 1e-6))))
    prev, t = s0, 1
    while t < n:
        m = min(B, n - t)
        pw = beta ** np.arange(1, m + 1)
        blk = pw * (prev + np.cumsum(a[t:t + m] / pw))
        out[t:t + m] = blk
        prev = blk[-1]
        t += m
    return out


def variance_path(r, mu, omega, alpha, beta, gamma, s0=None):
    e = r - mu
    e2 = e * e
    a = np.empty(len(r))
    a[0] = 0.0
    a[1:] = omega + (alpha + gamma * (e[:-1] < 0)) * e2[:-1]
    return _linrec(a, beta, float(np.mean(e2)) if s0 is None else s0)


def _loglik(r, mu, s2, dist, nu):
    e = r - mu
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return -np.inf
    if dist == "normal":
        return float(-0.5 * np.sum(math.log(2 * math.pi) + np.log(s2) + e * e / s2))
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return float(len(r) * c - 0.5 * np.sum(np.log(s2)) - (nu + 1) / 2 * np.sum(np.log1p(e * e / ((nu - 2) * s2))))


def _softmax_params(x, k):
    z = np.concatenate([x[:k], [0.0]])
    z -= z.max()
    w = np.exp(z)
    return w / w.sum()


def _unpack(x, gjr, dist):
    mu, omega = x[0], math.exp(min(max(x[1], -50), 50))
    k = 3 if gjr else 2
    w = _softmax_params(x[2:2 + k], k)
    alpha, beta = w[0], w[1]
    gamma = 2 * w[2] if gjr else 0.0
    nu = 2.1 + math.exp(min(max(x[2 + k], -50), 50)) if dist == "t" else None
    return mu, omega, alpha, beta, gamma, nu


def fit_garch(r, model: str = "garch", dist: str = "normal"):
    r = np.asarray(r, float)
    gjr = model == "gjr"
    var = float(np.var(r))
    k = 3 if gjr else 2

    def nll(x):
        mu, omega, alpha, beta, gamma, nu = _unpack(x, gjr, dist)
        if omega <= 0 or omega > 1e6 * var:
            return 1e12
        s2 = variance_path(r, mu, omega, alpha, beta, gamma)
        ll = _loglik(r, mu, s2, dist, nu)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95)):
        w = [a0, b0] + ([0.05] if gjr else [])
        slack = 1 - sum(w) + (0.0 if not gjr else 0.0)
        w_full = np.array(w + [max(slack, 0.01)])
        xs = np.log(w_full[:-1] / w_full[-1])
        x0 = np.concatenate([[r.mean(), math.log(var * max(1 - a0 - b0, 0.02))], xs, [math.log(6.0)] if dist == "t" else []])
        res = optim.minimize(nll, x0, maxiter=200)
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, beta, gamma, nu = _unpack(best.x, gjr, dist)
    s2 = variance_path(r, mu, omega, alpha, beta, gamma)
    ll = _loglik(r, mu, s2, dist, nu)
    kp = 4 + (1 if gjr else 0) + (1 if dist == "t" else 0)
    persist = alpha + beta + gamma / 2
    names = ["mu", "omega", "alpha", "beta"] + (["gamma"] if gjr else []) + (["nu"] if dist == "t" else [])
    vals = [mu, omega, alpha, beta] + ([gamma] if gjr else []) + ([nu] if dist == "t" else [])

    def nll_c(v):                                  # likelihood in constrained parameters, for the Hessian
        m_, o_, a_, b_ = v[0], v[1], v[2], v[3]
        g_ = v[4] if gjr else 0.0
        n_ = v[-1] if dist == "t" else None
        if o_ <= 0 or a_ < 0 or b_ < 0 or g_ < 0 or (n_ is not None and n_ <= 2.01):
            return 1e12
        ll_ = _loglik(r, m_, variance_path(r, m_, o_, a_, b_, g_), dist, n_)
        return -ll_ if np.isfinite(ll_) else 1e12
    try:
        cov = np.linalg.inv(optim.hessian(nll_c, np.array(vals), rel=2e-3))
        se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except Exception:
        se = np.full(len(vals), np.nan)
    z = (r - mu) / np.sqrt(s2)
    return {"model": model, "dist": dist, "mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma, "nu": nu, "loglik": ll, "aic": -2 * ll + 2 * kp,
            "bic": -2 * ll + kp * math.log(len(r)), "k": kp, "s2": s2, "z": z, "persistence": persist, "names": names, "values": vals, "se": se,
            "uncond_var": omega / (1 - persist) if persist < 1 else float("inf"), "half_life": (math.log(0.5) / math.log(persist)) if 0 < persist < 1 else float("inf")}


def next_variance(fit, r):
    e = r[-1] - fit["mu"]
    return fit["omega"] + (fit["alpha"] + fit["gamma"] * (e < 0)) * e * e + fit["beta"] * fit["s2"][-1]


def forecast_vol(fit, r, h):
    s1 = next_variance(fit, r)
    V, P = fit["uncond_var"], fit["persistence"]
    return np.sqrt(np.array([V + P ** k * (s1 - V) for k in range(h)])) if np.isfinite(V) else np.full(h, math.sqrt(s1))


def ewma_variance(r, lam=0.94):
    e2 = (r - r.mean()) ** 2
    a = np.empty(len(r))
    a[0] = 0.0
    a[1:] = (1 - lam) * e2[:-1]
    return _linrec(a, lam, float(e2[:30].mean()))


def vol_forecast_skill(r, spec, frac=0.3):
    """Fit on the first (1-frac), then compare one-step variance forecasts on the rest: GARCH vs EWMA(0.94) vs constant. Loss = QLIKE."""
    n = len(r)
    m = int(frac * n)
    if m < 100 or n - m < 400:
        return None
    train = r[:n - m]
    f = fit_garch(train, spec["model"], spec["dist"])
    s2 = variance_path(r, f["mu"], f["omega"], f["alpha"], f["beta"], f["gamma"], s0=float(np.mean((train - f["mu"]) ** 2)))[-m:]
    ew = ewma_variance(r)[-m:]
    const = np.full(m, float(np.var(train)))
    r2 = ((r[-m:] - f["mu"]) ** 2)
    ql = lambda s: np.log(s) + r2 / s
    lg, le, lc = ql(s2), ql(ew), ql(const)
    d = lg - le
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(m))) if d.std() > 0 else 0.0
    return {"m": m, "qlike_garch": float(lg.mean()), "qlike_ewma": float(le.mean()), "qlike_const": float(lc.mean()), "dm_t": t,
            "dm_p": float(2 * (1 - stats.norm_cdf(abs(t))))}


def garch(series: Series, max_obs: int = 4000, horizon: int = 20, models=None, cfg: Config | None = None, progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    dates, lr = series.log_returns()
    lr = np.asarray(lr, float) * 100.0
    ok = np.isfinite(lr)
    dates, lr = [d for d, k in zip(dates, ok) if k][-max_obs:], lr[ok][-max_obs:]
    if len(lr) < 300:
        raise ValueError(f"GARCH needs at least 300 returns; got {len(lr)}.")
    ppy = series.ppy
    specs = models or [("garch", "normal"), ("garch", "t"), ("gjr", "normal"), ("gjr", "t")]
    fits = []
    for i, (m, d) in enumerate(specs):
        progress(f"Fitting {m.upper()} ({d})", 10 + int(65 * i / len(specs)))
        fits.append(fit_garch(lr, m, d))
    fits.sort(key=lambda f: f["bic"])
    best = fits[0]
    lab = f"{'GJR-GARCH' if best['model'] == 'gjr' else 'GARCH'}(1,1) with {'Student-t' if best['dist'] == 't' else 'normal'} errors"
    ann = lambda s: float(math.sqrt(s * ppy))
    cur = math.sqrt(next_variance(best, lr))
    longrun = math.sqrt(best["uncond_var"]) if np.isfinite(best["uncond_var"]) else float("nan")
    hist = np.sqrt(best["s2"])
    pct_rank = float(np.mean(hist <= cur))
    sql = series.sql
    facts = []

    def add(type_, title, msg, sev, score, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=[series.name]))

    P = best["persistence"]
    hl = best["half_life"]
    add("q_garch_model", f"Selected {lab}", f"{lab} has the lowest BIC ({best['bic']:.1f}) of {len(fits)} candidates for {series.name}. Persistence alpha + beta"
        f"{' + gamma/2' if best['model'] == 'gjr' else ''} = {P:.3f}; volatility shocks decay with a half-life of "
        f"{('%.0f' % hl) if np.isfinite(hl) else 'infinite'} periods. Maximum likelihood on {len(lr):,} returns.", "info", 0.4, persistence=P, half_life=hl, bic=best["bic"],
        model=best["model"], dist=best["dist"])
    if np.isfinite(longrun):
        add("q_garch_vol", "Current vs long-run volatility", f"The model's next-period volatility is {ann(cur ** 2):.1f}% annualized versus a long-run level of {ann(longrun ** 2):.1f}% "
            f"({cur / longrun:.2f}x); current volatility is higher than {pct_rank:.0%} of the sample.", "notable" if cur / longrun >= 1.5 or cur / longrun <= 0.6 else "info",
            min(1.0, abs(math.log(cur / longrun))), current=ann(cur ** 2), longrun=ann(longrun ** 2), ratio=cur / longrun)
    if best["model"] == "gjr":
        i = best["names"].index("gamma")
        t = best["values"][i] / best["se"][i] if best["se"][i] and np.isfinite(best["se"][i]) else float("nan")
        if np.isfinite(t) and abs(t) > 2:
            add("q_garch_leverage", "Asymmetric (leverage) effect", f"Negative returns raise next-period variance more than positive ones (gamma = {best['gamma']:.3f}, z = {t:.1f}); "
                f"a symmetric GARCH would understate risk after sell-offs.", "info", 0.4, gamma=best["gamma"], z=float(t))
    if best["dist"] == "t":
        lln = next((f for f in fits if f["model"] == best["model"] and f["dist"] == "normal"), None)
        lr_p = stats.chi2_sf(2 * (best["loglik"] - lln["loglik"]), 1) if lln else float("nan")
        add("q_garch_tails", "Fat-tailed innovations", f"Student-t innovations fit better than normal (degrees of freedom {best['nu']:.1f}"
            f"{'' if not np.isfinite(lr_p) else f', likelihood-ratio p {lr_p:.1e}'}); Gaussian VaR would understate extreme losses.", "info", 0.3, nu=best["nu"])
    lb_z, lb_z2 = stats.ljung_box(best["z"], 10)[:2], stats.ljung_box(best["z"] ** 2, 10)[:2]
    alm0, alm1 = stats.arch_lm(lr, 5), stats.arch_lm(best["z"], 5)
    if alm0 and alm0[1] < 0.01:
        add("q_garch_clustering", "Volatility clustering", f"The raw returns show ARCH effects (ARCH-LM p = {alm0[1]:.1e}); after the GARCH filter the standardized residuals "
            f"{'still show' if alm1 and alm1[1] < 0.01 else 'show no remaining'} ARCH effects (p = {alm1[1]:.3f}).", "info" if alm1 and alm1[1] >= 0.01 else "notable", 0.4)
    if lb_z2[1] < 0.01:
        add("q_garch_resid", "Misspecified variance dynamics", f"Squared standardized residuals are still autocorrelated (Ljung-Box p = {lb_z2[1]:.1e}); the variance model is incomplete.", "notable", 0.5)
    skill = vol_forecast_skill(lr, {"model": best["model"], "dist": best["dist"]})
    if skill:
        win = skill["qlike_garch"] < skill["qlike_ewma"] and skill["dm_p"] < 0.05
        add("q_garch_skill", "Out-of-sample volatility forecast", f"Over the last {skill['m']} periods the model's mean QLIKE loss is {skill['qlike_garch']:.3f} versus {skill['qlike_ewma']:.3f} for "
            f"EWMA(0.94) and {skill['qlike_const']:.3f} for a constant variance (lower is better; Diebold-Mariano p vs EWMA = {skill['dm_p']:.3f}). " +
            ("GARCH forecasts volatility significantly better than EWMA." if win else "GARCH is not significantly better than the simple EWMA benchmark."), "info", 0.3,
            **{k: float(v) for k, v in skill.items()})
    for i, f in enumerate(facts, 1):
        f.id = f"G{i:03d}"
    fv = forecast_vol(best, lr, horizon)
    real = np.full(len(lr), np.nan)
    w = 21
    for i in range(w, len(lr)):
        real[i] = lr[i - w:i].std(ddof=1)
    rows = [(dates[i], ann(hist[i] ** 2), (ann(real[i] ** 2) if np.isfinite(real[i]) else ann(hist[i] ** 2))) for i in range(len(lr))][-1000:]
    spec = {"kind": "multiline", "names": ["GARCH conditional vol (% ann.)", "21-period realized vol (% ann.)"], "xlabel": "time", "ylabel": "annualized volatility, %"}
    param_rows = [[n, float(v), float(s)] for n, v, s in zip(best["names"], best["values"], best["se"])]
    cand = [[f"{'GJR' if f['model'] == 'gjr' else 'GARCH'}-{f['dist']}", f["loglik"], f["aic"], f["bic"], f["persistence"]] for f in fits]
    summary = {"series": series.name, "returns used": len(lr), "selected": lab, "log-likelihood": best["loglik"], "BIC": best["bic"], "persistence": P,
               "half-life (periods)": hl if np.isfinite(hl) else "inf", "long-run vol % (ann.)": ann(longrun ** 2) if np.isfinite(longrun) else "n/a", "next-period vol % (ann.)": ann(cur ** 2),
               "periods per year": ppy}
    out = SubReport(kind="garch", title=f"GARCH: {series.name}", summary=summary, facts=facts, setup_sql=[series.source.source_sql()], warnings=list(series.notes),
                    tables={"Parameters": (["parameter", "estimate", "std err"], param_rows), "Candidate models": (["model", "log-lik", "AIC", "BIC", "persistence"], cand),
                            "Volatility forecast": (["step", "vol % (ann.)"], [[i + 1, ann(v * v)] for i, v in enumerate(fv)])},
                    charts={"G001": (spec, rows)})
    out.provenance = provenance.build([provenance.input_record(series.source, "data", cfg.hash_data)], {"horizon": horizon, "max_obs": max_obs}, cfg, None, None, facts, command)
    progress("Done", 100)
    return out
