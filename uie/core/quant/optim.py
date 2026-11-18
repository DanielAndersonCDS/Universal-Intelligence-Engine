"""Small numerical optimisers (numpy only): BFGS with numeric gradients, Nelder-Mead, and a numeric Hessian."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class OptResult:
    x: np.ndarray
    fun: float
    nit: int
    success: bool


def _grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        step = h * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        g[i] = (f(xp) - f(xm)) / (2 * step)
    return g


def bfgs(f, x0, maxiter: int = 300, tol: float = 1e-8) -> OptResult:
    x = np.asarray(x0, float).copy()
    n = len(x)
    fx = f(x)
    if not np.isfinite(fx):
        return OptResult(x, fx, 0, False)
    H = np.eye(n)
    g = _grad(f, x)
    for it in range(1, maxiter + 1):
        if np.linalg.norm(g) < tol:
            return OptResult(x, fx, it, True)
        d = -H @ g
        if d @ g >= 0:
            H, d = np.eye(n), -g
        t, ok = 1.0, False
        for _ in range(40):                       # backtracking (Armijo)
            xn = x + t * d
            fn = f(xn)
            if np.isfinite(fn) and fn <= fx + 1e-4 * t * (g @ d):
                ok = True
                break
            t *= 0.5
        if not ok:
            return OptResult(x, fx, it, np.linalg.norm(g) < 1e-4)
        gn = _grad(f, xn)
        s, y = xn - x, gn - g
        sy = float(s @ y)
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        if abs(fx - fn) < 1e-12 * (1 + abs(fx)) and np.linalg.norm(gn) < 1e-5:
            x, fx, g = xn, fn, gn
            return OptResult(x, fx, it, True)
        x, fx, g = xn, fn, gn
    return OptResult(x, fx, maxiter, False)


def nelder_mead(f, x0, maxiter: int = 2000, tol: float = 1e-9) -> OptResult:
    x0 = np.asarray(x0, float)
    n = len(x0)
    sim = np.vstack([x0] + [x0 + np.eye(n)[i] * (0.1 * max(1.0, abs(x0[i]))) for i in range(n)])
    fs = np.array([f(p) for p in sim])
    for it in range(1, maxiter + 1):
        o = np.argsort(fs)
        sim, fs = sim[o], fs[o]
        if abs(fs[-1] - fs[0]) < tol * (1 + abs(fs[0])) and np.max(np.abs(sim[1:] - sim[0])) < 1e-7:
            return OptResult(sim[0], fs[0], it, True)
        c = sim[:-1].mean(0)
        xr = c + (c - sim[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - sim[-1])
            fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * ((xr if fr < fs[-1] else sim[-1]) - c)
            fc = f(xc)
            if fc < min(fr, fs[-1]):
                sim[-1], fs[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                fs[1:] = [f(p) for p in sim[1:]]
    o = np.argsort(fs)
    return OptResult(sim[o[0]], fs[o[0]], maxiter, False)


def minimize(f, x0, **kw) -> OptResult:
    """BFGS first; Nelder-Mead polish/fallback if BFGS stalls."""
    r = bfgs(f, x0, **kw)
    if not r.success or not np.isfinite(r.fun):
        r2 = nelder_mead(f, r.x if np.isfinite(r.fun) else x0)
        if np.isfinite(r2.fun) and (not np.isfinite(r.fun) or r2.fun < r.fun):
            r = r2
            r = bfgs(f, r.x, maxiter=50) if np.isfinite(r.fun) else r
    return r


def hessian(f, x, rel: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    h = np.array([rel * max(abs(v), 1e-2) for v in x])
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h[i]; xpp[j] += h[j]
            xpm[i] += h[i]; xpm[j] -= h[j]
            xmp[i] -= h[i]; xmp[j] += h[j]
            xmm[i] -= h[i]; xmm[j] -= h[j]
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h[i]; xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H
