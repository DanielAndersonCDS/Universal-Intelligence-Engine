"""ARIMA(p,d,q) by conditional sum of squares with stationarity/invertibility enforced through a partial-autocorrelation
parametrisation. Order chosen by AIC over a small grid, d from unit-root tests. Forecasts with prediction intervals, residual
diagnostics, and an out-of-sample comparison against a random-walk forecast (Diebold-Mariano)."""
from __future__ import annotations
import math
import numpy as np
from .. import provenance, stats
from ..config import Config
from ..facts import Fact
from ..subreport import SubReport
from . import optim
from .series import Series


def _pacf_to_poly(x):
    """Map unconstrained x to AR coefficients of a stationary polynomial (Jones/Monahan): phi(B)=1-sum phi_i B^i."""
    r = np.tanh(np.asarray(x, float))
    phi = np.array([])
    for k, rk in enumerate(r):
        phi = np.append(phi - rk * phi[::-1], rk) if k else np.array([rk])
    return phi


def _poly_to_pacf(phi):
    phi = np.array(phi, float)
    out = []
    for k in range(len(phi), 0, -1):
        rk = phi[-1]
        out.append(rk)
        if k > 1:
            phi = (phi[:-1] + rk * phi[:-1][::-1]) / (1 - rk * rk)
    return np.arctanh(np.clip(np.array(out[::-1]), -0.999, 0.999))


def _residuals(z, phi, theta):
    n, p, q = len(z), len(phi), len(theta)
    u = z.copy()
    for i, a in enumerate(phi, 1):
        u[i:] -= a * z[:-i]
    u[:p] = 0.0
    if not q:
        return u
    e = np.zeros(n)
    ul, th = u.tolist(), list(theta)
    el = [0.0] * n
    for t in range(p, n):
        v = ul[t]
        for j in range(1, q + 1):
            if t - j >= 0:
                v -= th[j - 1] * el[t - j]
        el[t] = v
    return np.array(el)


def _unpack(params, p, q, mean):
    mu = params[0] if mean else 0.0
    off = 1 if mean else 0
    phi = _pacf_to_poly(params[off:off + p]) if p else np.array([])
    theta = -_pacf_to_poly(params[off + p:off + p + q]) if q else np.array([])
    return mu, phi, theta


def fit_arma(y, p: int, q: int, mean: bool = True):
    """CSS fit of ARMA(p,q) to y (already differenced). Returns dict or None."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 30 + 5 * (p + q):
        return None
    start = np.concatenate([[y.mean()] if mean else [], np.full(p, 0.05), np.full(q, 0.05)])

    def sse(params):
        mu, phi, theta = _unpack(params, p, q, mean)
        e = _residuals(y - mu, phi, theta)[p:]
        v = float(e @ e)
        return v if np.isfinite(v) else 1e300

    best = None
    for init in (start, np.concatenate([[y.mean()] if mean else [], np.full(p, -0.3), np.full(q, -0.3)])):
        res = optim.minimize(sse, init, maxiter=150)
        if best is None or res.fun < best.fun:
            best = res
    mu, phi, theta = _unpack(best.x, p, q, mean)
    e = _residuals(y - mu, phi, theta)[p:]
    ne = len(e)
    s2 = float(e @ e) / ne
    k = int(mean) + p + q
    aic = ne * math.log(s2) + 2 * (k + 1)
    bic = ne * math.log(s2) + (k + 1) * math.log(ne)
    return {"p": p, "q": q, "mu": float(mu), "phi": phi, "theta": theta, "sigma2": s2, "aic": aic, "bic": bic, "nobs": ne, "resid": e, "k": k, "sse": float(e @ e),
            "params_flat": np.concatenate([[mu] if mean else [], phi, theta]), "mean": mean}


def _param_se(y, fit):
    """Standard errors from the numeric Hessian of the Gaussian CSS log-likelihood in (mu, phi, theta)."""
    p, q, mean, n = fit["p"], fit["q"], fit["mean"], fit["nobs"]
    flat = fit["params_flat"]

    def nll(v):
        off = 1 if mean else 0
        mu = v[0] if mean else 0.0
        phi, theta = v[off:off + p], v[off + p:]
        e = _residuals(y - mu, phi, theta)[p:]
        return 0.5 * n * math.log(max(float(e @ e) / n, 1e-300)) + 0.5 * n
    try:
        H = optim.hessian(nll, flat)
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        return se if np.all(np.isfinite(se)) else np.full(len(flat), np.nan)
    except Exception:
        return np.full(len(flat), np.nan)


def _integrate_poly(phi, d):
    """Coefficients c of Phi(B) = phi(B) (1-B)^d written as y_t = sum_i c_i y_{t-i} + ..."""
    poly = np.concatenate([[1.0], -np.asarray(phi, float)])
    for _ in range(d):
        poly = np.convolve(poly, [1.0, -1.0])
    return -poly[1:]


def psi_weights(phi, theta, d, h):
    big = _integrate_poly(phi, d)
    psi = [1.0]
    for j in range(1, h):
        v = theta[j - 1] if j <= len(theta) else 0.0
        for i in range(1, min(j, len(big)) + 1):
            v += big[i - 1] * psi[j - i]
        psi.append(v)
    return np.array(psi)


def forecast(y_levels, fit, d, h):
    """Point forecasts of the level series and 95% prediction intervals."""
    y = np.asarray(y_levels, float)
    z = y.copy()
    heads = []
    for _ in range(d):
        heads.append(z[-1])
        z = np.diff(z)
    zc = z - fit["mu"]
    e = _residuals(zc, fit["phi"], fit["theta"])
    ext, ee = list(zc), list(e)
    p, q = len(fit["phi"]), len(fit["theta"])
    fz = []
    for _ in range(h):
        v = sum(fit["phi"][i] * ext[-1 - i] for i in range(p)) + sum(fit["theta"][j] * ee[-1 - j] for j in range(q))
        ext.append(v)
        ee.append(0.0)
        fz.append(v + fit["mu"])
    f = np.array(fz)
    for last in reversed(heads):
        f = last + np.cumsum(f)
    psi = psi_weights(fit["phi"], fit["theta"], d, h)
    se = math.sqrt(fit["sigma2"]) * np.sqrt(np.cumsum(psi ** 2))
    return f, f - 1.96 * se, f + 1.96 * se


def choose_d(y, max_d=2):
    d, z, notes = 0, np.asarray(y, float), []
    while d < max_d:
        a = stats.adf(z, "c")
        if a is None or a.reject5:
            break
        notes.append(f"ADF on {'levels' if d == 0 else 'differences' * 1}: stat {a.stat:.2f}, p {a.p:.3f}: unit root not rejected")
        z = np.diff(z)
        d += 1
    return d, notes


def holdout_skill(y, d, p, q, mean, frac=0.2):
    """Fit on the first (1-frac) of the data, then compare one-step forecast errors on the rest with a random walk (Diebold-Mariano)."""
    n = len(y)
    m = max(20, int(frac * n))
    if n - m < 80 + 5 * (p + q):
        return None
    train = y[:n - m]
    z = np.diff(train, n=d) if d else train
    fit = fit_arma(z, p, q, mean)
    if fit is None:
        return None
    zall = np.diff(y, n=d) if d else y
    e = _residuals(zall - fit["mu"], fit["phi"], fit["theta"])
    e_test = e[-m:]
    naive = np.diff(y)[-m:] if d >= 1 else (y[1:] - y[:-1])[-m:]
    if d == 0:
        naive = (y[-m:] - np.concatenate([[y[-m - 1]], y[-m:-1]]))
    dl = e_test ** 2 - naive ** 2
    sd = dl.std(ddof=1)
    t = float(dl.mean() / (sd / math.sqrt(m))) if sd > 0 else 0.0
    return {"m": m, "rmse_model": float(math.sqrt(np.mean(e_test ** 2))), "rmse_naive": float(math.sqrt(np.mean(naive ** 2))), "dm_t": t,
            "dm_p": float(2 * (1 - stats.norm_cdf(abs(t))))}


def arima(series: Series, order=None, max_p: int = 2, max_q: int = 2, horizon: int = 20, log: bool = False, cfg: Config | None = None,
          progress=lambda s, p=0: None, command=None) -> SubReport:
    cfg = cfg or Config()
    y = np.asarray(series.values, float)
    if log:
        if np.any(y <= 0):
            raise ValueError("Cannot take logs of non-positive values.")
        y = np.log(y)
    y = y[-2500:]
    dates = series.dates[-len(y):]
    notes = list(series.notes)
    if order:
        p0, d, q0 = order
        grid = [(p0, q0)]
    else:
        d, dn = choose_d(y)
        notes += dn
        grid = [(p, q) for p in range(max_p + 1) for q in range(max_q + 1)]
    z = np.diff(y, n=d) if d else y
    mean = d < 2
    fits = []
    for i, (p, q) in enumerate(grid):
        progress(f"Fitting ARIMA({p},{d},{q})", 10 + int(60 * i / len(grid)))
        f = fit_arma(z, p, q, mean)
        if f:
            fits.append(f)
    if not fits:
        raise ValueError("Series too short to fit an ARIMA model.")
    fits.sort(key=lambda f: f["aic"])
    best = fits[0]
    p, q = best["p"], best["q"]
    se = _param_se(z, best)
    names = (["mean"] if mean else []) + [f"AR{i}" for i in range(1, p + 1)] + [f"MA{j}" for j in range(1, q + 1)]
    flat = best["params_flat"]
    coef_rows = []
    for nm, v, s_ in zip(names, flat, se):
        t = v / s_ if s_ and np.isfinite(s_) and s_ > 0 else float("nan")
        coef_rows.append([nm, float(v), float(s_), float(t), float(2 * (1 - stats.norm_cdf(abs(t)))) if np.isfinite(t) else float("nan")])
    res = best["resid"]
    lags = min(20, max(5, len(res) // 10))
    q_stat, q_p, lags = stats.ljung_box(res, lags, p + q)
    jb = stats.jarque_bera(res)
    alm = stats.arch_lm(res, 5)
    skill = holdout_skill(y, d, p, q, mean)
    progress("Forecasting", 85)
    f, lo, hi = forecast(y, best, d, horizon)
    lvl = (lambda v: np.exp(v)) if log else (lambda v: v)
    step = dates[-1] - dates[-2]
    fdates = [dates[-1] + step * (i + 1) for i in range(horizon)] if hasattr(step, "days") else list(range(horizon))
    name = f"ARIMA({p},{d},{q})"
    sql = series.sql
    facts = []

    def add(type_, title, msg, sev, score, **m):
        facts.append(Fact(type=type_, title=title, metrics={"message": msg, **m}, severity=sev, score=score, evidence_sql=sql, columns=[series.name]))

    add("q_arima_model", f"Selected {name}", f"{name} has the lowest AIC ({best['aic']:.1f}) among {len(fits)} candidate orders for {series.name}"
        f"{' (log scale)' if log else ''}; differencing order d = {d} chosen by ADF tests. Fitted by conditional least squares on {best['nobs']:,} observations.", "info", 0.4,
        order=[p, d, q], aic=best["aic"], bic=best["bic"])
    fm = f"{lvl(f[-1]):,.4g}"
    add("q_arima_forecast", f"{horizon}-step forecast", f"The {horizon}-step-ahead forecast of {series.name} is {fm} with a 95% prediction interval of "
        f"{lvl(lo[-1]):,.4g} to {lvl(hi[-1]):,.4g} (last observed {lvl(y[-1]):,.4g}). Intervals assume the fitted model and Gaussian errors; they widen quickly and are model-based, not guaranteed.",
        "info", 0.3, horizon=horizon, point=float(lvl(f[-1])), lo=float(lvl(lo[-1])), hi=float(lvl(hi[-1])))
    if skill:
        beats = skill["dm_p"] < 0.05 and skill["rmse_model"] < skill["rmse_naive"]
        add("q_arima_skill", "Out-of-sample skill vs random walk",
            f"On the last {skill['m']} observations the model's one-step RMSE is {skill['rmse_model']:.4g} versus {skill['rmse_naive']:.4g} for a random-walk forecast "
            f"(Diebold-Mariano p = {skill['dm_p']:.3f}). " + ("The model forecasts significantly better than a random walk." if beats else
            "The model does not forecast significantly better than a random walk, which is typical for financial prices; do not rely on it for point forecasts."),
            "info" if beats else "notable", 0.6 if not beats else 0.3, **{k: float(v) for k, v in skill.items()})
    if q_p < 0.05:
        add("q_arima_resid", "Residual autocorrelation", f"Ljung-Box on the residuals rejects white noise (Q({lags}) = {q_stat:.1f}, p = {q_p:.3f}); the model leaves structure unexplained, "
            f"so consider higher orders or other effects.", "notable", 0.5, q=q_stat, p=q_p)
    if alm and alm[1] < 0.01:
        add("q_arima_arch", "Volatility clustering in residuals", f"The squared residuals are autocorrelated (ARCH-LM p = {alm[1]:.1e}): the variance is not constant, so the "
            f"prediction intervals are unreliable. A GARCH model for the variance is appropriate.", "notable", 0.6, lm=alm[0], p=alm[1])
    if jb and jb["p"] < 0.001 and jb["kurtosis"] > 1:
        add("q_arima_tails", "Fat-tailed residuals", f"Residuals have excess kurtosis {jb['kurtosis']:.1f} (Jarque-Bera p {'< 0.0001' if jb['p'] < 1e-4 else format(jb['p'], '.4f')}); "
            f"Gaussian prediction intervals understate extreme moves.", "info", 0.3, kurtosis=jb["kurtosis"])
    for i, ft in enumerate(facts, 1):
        ft.id = f"A{i:03d}"
    tail = min(len(y), 250)
    rows = [(dates[-tail + i], float(lvl(y[-tail + i])), None, None) for i in range(tail)] + [(fdates[i], float(lvl(f[i])), float(lvl(lo[i])), float(lvl(hi[i]))) for i in range(horizon)]
    rows = [(r[0], r[1], r[2] if r[2] is not None else r[1], r[3] if r[3] is not None else r[1]) for r in rows]
    spec = {"kind": "multiline", "names": [series.name, "95% lower", "95% upper"], "band": [1, 2], "xlabel": "time", "ylabel": series.name}
    cand = [[f"ARIMA({x['p']},{d},{x['q']})", x["aic"], x["bic"], x["sigma2"], x["nobs"]] for x in fits]
    summary = {"series": series.name, "observations": len(y), "model": name, "differencing d": d, "AIC": best["aic"], "BIC": best["bic"], "residual variance": best["sigma2"],
               f"Ljung-Box p (lags {lags})": q_p, "periods per year": series.ppy}
    out = SubReport(kind="arima", title=f"ARIMA: {series.name}", summary=summary, facts=facts, setup_sql=[series.source.source_sql()], warnings=notes[:1] + series.notes,
                    tables={"Coefficients": (["term", "estimate", "std err", "z", "p"], coef_rows), "Candidate orders": (["model", "AIC", "BIC", "sigma2", "n"], cand),
                            "Forecast": (["date", "forecast", "lower 95%", "upper 95%"], [[fdates[i], float(lvl(f[i])), float(lvl(lo[i])), float(lvl(hi[i]))] for i in range(horizon)])},
                    charts={"A002": (spec, rows)})
    out.provenance = provenance.build([provenance.input_record(series.source, "data", cfg.hash_data)], {"order": [p, d, q], "log": log, "horizon": horizon, "max_p": max_p, "max_q": max_q},
                                      cfg, None, None, facts, command)
    progress("Done", 100)
    return out
