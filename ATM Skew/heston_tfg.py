"""
TFG - Option pricing under stochastic volatility (Heston).  Shared library.

Everything is written in "forward" form: for each maturity T the forward
F = S0 exp((r-q)T) is estimated from put-call parity, so dividends do not need to
be modelled explicitly.  The call price is  C = exp(-rT) * Heston(F, K, T, r=0).

Main functions
  heston_call_fwd       Heston price, Fourier inversion (Little Heston Trap), vectorised over strikes
  decomp_call_fwd       Alos (2012) decomposition-formula approximation, closed form, vectorised
  implied_vol_fwd       Black-76 implied volatility
  load_market           download (or reload today's cached) option chain from Yahoo Finance
  build_clean_chain     forward by parity, OTM options only, quality filters, implied vols
  synthetic_chain       fake market generated from a known Heston -> to test the code
  calibrate             least squares in vega-weighted price space (~ implied-vol error)
  market_atm_skew       ATM implied-vol level, skew and curvature per maturity (local quadratic fit)
  model_atm_skew        the same quantities implied by a model (Fourier or decomposition)
  fit_power_law         skew ~ c * T^(-alpha) by log-log OLS
"""
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.optimize import brentq, least_squares


# ----------------------------------------------------------------------------
# Black-76 (Black-Scholes written on the forward)
# ----------------------------------------------------------------------------
def bs_call_fwd(F, K, T, sigma, r):
    F, K, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(sigma, float)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.exp(-r * T) * (F * ss.norm.cdf(d1) - K * ss.norm.cdf(d1 - sd))


def bs_vega_fwd(F, K, T, sigma, r):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.exp(-r * T) * F * ss.norm.pdf(d1) * np.sqrt(T)


def implied_vol_fwd(price, F, K, T, r):
    """Scalar implied vol; returns nan if the price violates no-arbitrage bounds."""
    lo = np.exp(-r * T) * max(F - K, 0.0)
    hi = np.exp(-r * T) * F
    if not (lo + 1e-12 * F < price < hi - 1e-12 * F):
        return np.nan
    try:
        return brentq(lambda s: bs_call_fwd(F, K, T, s, r) - price, 1e-4, 5.0, xtol=1e-12)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# Heston (Little Heston Trap) - vectorised over strikes
# ----------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_GRIDS = {}


def _u_grid(T):
    """Integration grid adapted to the maturity: short maturities need a longer
    u-range because the characteristic function decays like exp(-sigma^2 T u^2 / 2)."""
    upper = float(np.clip(12.0 / (0.08 * np.sqrt(T)), 200.0, 4000.0))
    upper = float(np.round(upper, -1))
    if upper not in _GRIDS:
        panels = int(np.ceil(upper / 5.0))
        edges = np.linspace(0.0, upper, panels + 1)
        a, b = edges[:-1, None], edges[1:, None]
        u = (0.5 * (b - a) * _GL_X[None, :] + 0.5 * (a + b)).ravel()
        w = (0.5 * (b - a) * _GL_W[None, :]).ravel()
        _GRIDS[upper] = (u, w)
    return _GRIDS[upper]


def _cf(u, T, kappa, theta, nu, rho, v0, j):
    """log-forward characteristic function with x0 = 0 and r = 0 (Little Heston Trap)."""
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho * nu if j == 1 else kappa
    iu = 1j * u
    d = np.sqrt((rho * nu * iu - bj) ** 2 - nu**2 * (2 * uj * iu - u**2))
    g = (bj - rho * nu * iu - d) / (bj - rho * nu * iu + d)
    e = np.exp(-d * T)
    C = (kappa * theta / nu**2) * ((bj - rho * nu * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = ((bj - rho * nu * iu - d) / nu**2) * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v0)


def heston_call_fwd(F, K, T, r, kappa, theta, nu, rho, v0):
    """European call(s) for strikes K (array) on forward F, maturity T (Fourier inversion)."""
    K = np.atleast_1d(np.asarray(K, float))
    U, W = _u_grid(T)
    k = np.log(K / F)[:, None]  # log-moneyness
    u = U[None, :]
    out = []
    for j in (1, 2):
        phi = _cf(u, T, kappa, theta, nu, rho, v0, j)
        integrand = np.real(np.exp(-1j * u * k) * phi / (1j * u))
        out.append(0.5 + (integrand * W[None, :]).sum(axis=1) / np.pi)
    P1, P2 = out
    return np.exp(-r * T) * (F * P1 - K * P2)


# ----------------------------------------------------------------------------
# Decomposition formula (Alos 2012) - closed form, vectorised over strikes
# ----------------------------------------------------------------------------
def decomp_terms(T, kappa, theta, nu, v0):
    """Expected integrated variance, E[int sigma d<M,B>] and E[int d<M,M>] in closed form
    (formulas on p. 9 of the chapter, evaluated at t = 0)."""
    e1, e2 = np.exp(-kappa * T), np.exp(-2 * kappa * T)
    int_var = theta * T + (v0 - theta) / kappa * (1 - e1)
    qv = (nu**2 / kappa**2) * (
        theta * T + (v0 - theta) / kappa * (1 - e1) - 2 * theta / kappa * (1 - e1)
        - 2 * (v0 - theta) * T * e1 + theta / (2 * kappa) * (1 - e2) + (v0 - theta) / kappa * (e1 - e2)
    )
    cov = (nu / kappa**2) * (
        theta * kappa * T - 2 * theta + v0 + e1 * (2 * theta - v0) - kappa * T * e1 * (v0 - theta)
    )
    return int_var, cov, qv


def decomp_call_fwd(F, K, T, r, kappa, theta, nu, rho, v0):
    """V ~ BS(vbar) + rho/2 * H * E[int sigma d<M,B>] + 1/8 * K * E[int d<M,M>]."""
    K = np.atleast_1d(np.asarray(K, float))
    int_var, cov, qv = decomp_terms(T, kappa, theta, nu, v0)
    sd = np.sqrt(int_var)  # vbar * sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    pref = F * ss.norm.pdf(d1) / sd
    H = pref * (1.0 - d1 / sd)
    Kf = pref * (d1**2 - sd * d1 - 1.0) / sd**2
    bs = F * ss.norm.cdf(d1) - K * ss.norm.cdf(d1 - sd)
    return np.exp(-r * T) * (bs + 0.5 * rho * H * cov + 0.125 * Kf * qv)


PRICERS = {"fourier": heston_call_fwd, "decomposition": decomp_call_fwd}


# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------
TARGET_DAYS = (2, 4, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)


def fetch_yfinance_chain(ticker="^SPX", min_days=14, max_days=400, max_expiries=8, target_days=None):
    """Download an option chain (needs internet).  Returns (spot, DataFrame).
    Maturities are measured from the last trading session (the date of the quotes).
    If target_days is given, the expiries closest to those horizons are used."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")["Close"].dropna()
    spot = float(hist.iloc[-1])
    asof = hist.index[-1].date()
    exps = [(e, (dt.date.fromisoformat(e) - asof).days) for e in tk.options]
    exps = [(e, d) for e, d in exps if 1 <= d <= max_days]
    if target_days is not None:
        chosen = sorted({min(exps, key=lambda ed: abs(ed[1] - t))[0] for t in target_days})
    else:
        chosen = [e for e, d in exps if d >= min_days]
        if len(chosen) > max_expiries:
            idx = np.unique(np.linspace(0, len(chosen) - 1, max_expiries).round().astype(int))
            chosen = [chosen[i] for i in idx]
    cols = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "lastTradeDate"]
    frames = []
    for e in chosen:
        ch = tk.option_chain(e)
        for kind, d in (("call", ch.calls), ("put", ch.puts)):
            d = d[[c for c in cols if c in d.columns]].copy()
            if "contractSymbol" in d.columns and len(d):
                # ^SPX lists two roots for some dates (SPX AM-settled, SPXW PM-settled): keep the most common one
                root = d["contractSymbol"].str[:-15]
                d = d[root == root.value_counts().idxmax()]
            d["type"], d["expiry"] = kind, e
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["T"] = [(dt.date.fromisoformat(e) - asof).days / 365.0 for e in df["expiry"]]
    df["spot"], df["asof"] = spot, str(asof)
    return spot, df


def load_market(tickers=("^SPX", "SPY"), cache_dir="data", target_days=TARGET_DAYS, max_days=400):
    """Today's chain from the cache if it exists, otherwise download it and save it.
    Both option notebooks call this, so the data are downloaded only once a day."""
    os.makedirs(cache_dir, exist_ok=True)
    errors = []
    for t in tickers:
        path = os.path.join(cache_dir, f"{t.replace('^', '')}_{dt.date.today()}.csv")
        if os.path.exists(path):
            raw = pd.read_csv(path)
            return t, float(raw["spot"].iloc[0]), raw
        try:
            spot, raw = fetch_yfinance_chain(t, max_days=max_days, target_days=target_days)
            if len(raw) == 0:
                raise ValueError("empty chain")
            raw.to_csv(path, index=False)
            return t, spot, raw
        except Exception as e:  # try the next ticker
            errors.append(f"{t}: {repr(e)[:150]}")
    raise RuntimeError("could not download any option chain -> " + " | ".join(errors))


def fetch_risk_free(default=0.04):
    """13-week US T-bill yield (^IRX on Yahoo, quoted in %).  Falls back to `default`."""
    try:
        import yfinance as yf

        return float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100.0
    except Exception:
        return default


def _quote(q, max_rel_spread, min_price):
    """Price of one quote: bid-ask mid if there is a valid quote, otherwise the last trade.
    When the market is closed Yahoo often shows bid = ask = 0, hence the fallback."""
    bid = float(np.nan_to_num(q.get("bid", 0.0)))
    ask = float(np.nan_to_num(q.get("ask", 0.0)))
    last = float(np.nan_to_num(q.get("lastPrice", 0.0)))
    if bid > 0 and ask > bid:
        mid = 0.5 * (bid + ask)
        if mid >= min_price and (ask - bid) / mid <= max_rel_spread:
            return mid, "mid"
        return np.nan, None
    if last >= min_price and not q.get("_stale", False):
        return last, "last"
    return np.nan, None


def build_clean_chain(df, spot, r=0.04, max_abs_logm=0.25, max_rel_spread=0.5, min_price=0.05,
                      prefer_mid=True, min_rel_vega=0.02, outliers=True):
    """Forward from put-call parity, OTM options only, implied vols, then quality filters (clean_quotes)."""
    df = df.copy()
    if "contractSymbol" in df.columns and len(df):
        # ^SPX lists two series for some expiries (SPX AM-settled, SPXW PM-settled): keep the most common one
        root = df["contractSymbol"].astype(str).str[:-15]
        main_root = root.groupby([df["expiry"], df["type"]]).transform(lambda s: s.value_counts().idxmax())
        df = df[root == main_root]
    df["_stale"] = False
    if "lastTradeDate" in df.columns:  # last trades older than ~1 session are discarded
        ltd = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")
        df["_stale"] = (ltd < ltd.max() - pd.Timedelta(days=1)).fillna(True)
    rows = []
    for e, g in df.groupby("expiry"):
        T = g["T"].iloc[0]
        c = g[g.type == "call"].drop_duplicates("strike").set_index("strike")
        p = g[g.type == "put"].drop_duplicates("strike").set_index("strike")
        common = c.index.intersection(p.index)
        cpx = {K: _quote(c.loc[K], max_rel_spread, min_price) for K in common}
        ppx = {K: _quote(p.loc[K], max_rel_spread, min_price) for K in common}
        both = [K for K in common if cpx[K][1] and ppx[K][1]]
        if len(both) < 3:
            continue
        near = sorted(both, key=lambda K: abs(K - spot))[:3]
        F = float(np.mean([K + np.exp(r * T) * (cpx[K][0] - ppx[K][0]) for K in near]))
        for K in common:
            kind = "call" if K >= F else "put"
            px, src = (cpx if kind == "call" else ppx)[K]
            if src is None or abs(np.log(K / F)) > max_abs_logm:
                continue
            call_px = px if kind == "call" else px + np.exp(-r * T) * (F - K)  # put-call parity
            iv = implied_vol_fwd(call_px, F, K, T, r)
            if np.isnan(iv):
                continue
            q = (c if kind == "call" else p).loc[K]
            rows.append(dict(expiry=e, T=T, F=F, K=K, kind=kind, price=px, src=src, call_mid=call_px,
                             iv=iv, vega=bs_vega_fwd(F, K, T, iv, r), volume=q.get("volume", np.nan)))
    if not rows:
        raise ValueError("No usable option quotes after cleaning (market closed and no recent trades?)")
    out = pd.DataFrame(rows).sort_values(["T", "K"]).reset_index(drop=True)
    return clean_quotes(out, prefer_mid=prefer_mid, min_rel_vega=min_rel_vega, outliers=outliers)


def clean_quotes(chain, prefer_mid=True, min_rel_vega=0.02, outliers=True, min_dev=0.01, n_mad=4.0, window=9):
    """Quality filters applied after computing implied volatilities.
    1) prefer_mid: if most quotes have a live bid-ask, drop the few priced with the last trade
       (stale, non-synchronous prices create spikes in the smiles).
    2) min_rel_vega: drop options whose vega is below 2% of the largest vega of their maturity
       (deep out-of-the-money, very short maturities: the price is a few ticks and the implied
       volatility is not reliable).
    3) outliers (spikes): per maturity, remove the smooth shape of the smile with a degree-4
       polynomial in k, then compare each residual with the rolling median of the residuals of
       the 9 closest strikes; drop it if it deviates by more than max(1 vol point, 4 robust
       standard deviations). Works for isolated spikes and for pairs of adjacent bad quotes.
    The number of options removed by each step is stored in chain.attrs["cleaning"]."""
    info = {"options_before": len(chain)}
    if prefer_mid and (chain["src"] == "mid").mean() >= 0.5:
        info["last_price_quotes"] = int((chain["src"] == "last").sum())
        chain = chain[chain["src"] != "last"]
    if min_rel_vega:
        rel = chain["vega"] / chain.groupby("T")["vega"].transform("max")
        info["low_vega"] = int((rel < min_rel_vega).sum())
        chain = chain[rel >= min_rel_vega]
    chain = chain.sort_values(["T", "K"]).reset_index(drop=True)
    if outliers:
        keep = np.ones(len(chain), bool)
        for T, idx in chain.groupby("T").groups.items():
            pos = np.asarray(chain.index.get_indexer(idx))    # positions sorted by strike
            if len(pos) < 7:
                continue
            k = np.log(chain["K"].values[pos] / chain["F"].values[pos])
            iv = chain["iv"].values[pos]
            res = iv - np.polyval(np.polyfit(k, iv, min(4, len(k) - 3)), k)
            med = pd.Series(res).rolling(window, center=True, min_periods=3).median().values
            dev = np.abs(res - med)
            thr = max(min_dev, n_mad * 1.4826 * float(np.median(dev)))
            keep[pos[dev > thr]] = False
        info["iv_outliers"] = int((~keep).sum())
        chain = chain[keep].reset_index(drop=True)
    info["options_after"] = len(chain)
    chain.attrs["cleaning"] = info
    return chain


def synthetic_chain(true_params, spot=500.0, r=0.04, q=0.012,
                    maturities=(0.08, 0.25, 0.5, 1.0), noise_vol=0.003, seed=0, n_strikes=25):
    """Fake market generated from a known Heston, to test the code."""
    rng = np.random.default_rng(seed)
    kappa, theta, nu, rho, v0 = true_params
    rows = []
    for T in maturities:
        F = spot * np.exp((r - q) * T)
        width = min(0.25, 2.5 * np.sqrt(v0 * T) + 0.01)
        Ks = F * np.exp(np.linspace(-width, 0.6 * width, n_strikes))
        px = heston_call_fwd(F, Ks, T, r, kappa, theta, nu, rho, v0)
        for K, c in zip(Ks, px):
            iv = implied_vol_fwd(c, F, K, T, r)
            if np.isnan(iv):
                continue
            iv += noise_vol * rng.standard_normal()
            cm = bs_call_fwd(F, K, T, iv, r)
            rows.append(dict(expiry=f"T={T:.4f}", T=T, F=F, K=K, kind="call" if K >= F else "put",
                             price=cm, src="synthetic", call_mid=cm, iv=iv,
                             vega=bs_vega_fwd(F, K, T, iv, r), volume=np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
PARAM_NAMES = ["kappa", "theta", "nu", "rho", "v0"]
LOWER = np.array([0.05, 0.005, 0.05, -0.99, 0.0005])
UPPER = np.array([20.0, 1.0, 5.0, 0.99, 1.0])


def model_prices(p, chain, r, pricer="fourier"):
    f = PRICERS[pricer]
    kappa, theta, nu, rho, v0 = p
    out = np.empty(len(chain))
    for T, idx in chain.groupby("T").groups.items():
        sub = chain.loc[idx]
        out[chain.index.get_indexer(idx)] = f(sub["F"].iloc[0], sub["K"].values, T, r, kappa, theta, nu, rho, v0)
    return out


def model_ivs(p, chain, r, pricer="fourier"):
    m = model_prices(p, chain, r, pricer)
    return np.array([implied_vol_fwd(x, F, K, T, r) for x, F, K, T in zip(m, chain["F"], chain["K"], chain["T"])])


def calibrate(chain, r, pricer="fourier", starts=None, verbose=True, return_info=False):
    """Minimise sum(((model - market)/vega)^2): price error divided by vega ~ implied-vol error."""
    mkt = chain["call_mid"].values
    vega = np.maximum(chain["vega"].values, 1e-8 * chain["F"].values)

    def resid(p):
        return (model_prices(p, chain, r, pricer) - mkt) / vega

    if starts is None:
        v_atm = float(np.clip(chain.loc[chain["T"].idxmin(), "iv"] ** 2, 0.001, 0.5))
        starts = [[2.0, 0.04, 0.5, -0.7, v_atm], [1.0, 0.06, 0.3, -0.5, v_atm], [5.0, 0.04, 1.0, -0.8, v_atm]]
    best, nfev, t0 = None, 0, time.perf_counter()
    for s in starts:
        s = np.clip(s, LOWER + 1e-6, UPPER - 1e-6)
        sol = least_squares(resid, s, bounds=(LOWER, UPPER), x_scale=[1, 0.05, 0.3, 0.3, 0.05])
        nfev += sol.nfev
        if verbose:
            print(f"[{pricer}] start {np.round(s, 3)} -> cost {sol.cost:.6f}  params {np.round(sol.x, 4)}")
        if best is None or sol.cost < best.cost:
            best = sol
    seconds = time.perf_counter() - t0
    p = best.x
    fit = chain.copy()
    fit["model"] = model_prices(p, chain, r, pricer)
    fit["model_iv"] = model_ivs(p, chain, r, pricer)
    rmse_iv = float(np.sqrt(np.nanmean((fit["model_iv"] - fit["iv"]) ** 2)))
    params = dict(zip(PARAM_NAMES, map(float, p)))
    if return_info:
        return params, fit, rmse_iv, dict(seconds=seconds, nfev=nfev, starts=len(starts))
    return params, fit, rmse_iv


def feller(p):
    return 2 * p["kappa"] * p["theta"] - p["nu"] ** 2


# ----------------------------------------------------------------------------
# ATM implied-volatility skew
# ----------------------------------------------------------------------------
def market_atm_skew(chain, min_points=6, width=0.3):
    """For each maturity, fit iv = a + b k + c k^2 on |k| <= width * atm_vol * sqrt(T) (k = log(K/F)).
    a = ATM level, b = ATM skew (d iv / d k at k = 0), 2c = curvature."""
    rows = []
    for T, g in chain.groupby("T"):
        g = g.sort_values("K")
        k, iv = np.log(g["K"] / g["F"]).values, g["iv"].values
        if len(k) < min_points or k.min() > 0 or k.max() < 0:
            continue
        atm = float(np.interp(0.0, k, iv))
        band = max(0.01, width * atm * np.sqrt(T))
        m = np.abs(k) <= band
        if m.sum() < min_points:  # widen until there are enough points
            m = np.argsort(np.abs(k))[:min_points]
        coef, cov = np.polyfit(k[m], iv[m], 2, cov=True)
        rows.append(dict(T=T, days=round(T * 365), n=int(np.size(k[m])), atm_iv=coef[2], atm_skew=coef[1],
                         atm_skew_se=float(np.sqrt(cov[1, 1])), curvature=2 * coef[0]))
    return pd.DataFrame(rows)


def model_atm_skew(p, Ts, pricer="fourier", h=0.002):
    """ATM level and skew implied by a model, by central differences in log-strike."""
    f = PRICERS[pricer]
    rows = []
    for T in np.atleast_1d(Ts):
        K = np.exp(np.array([-h, 0.0, h]))
        c = f(1.0, K, T, 0.0, p["kappa"], p["theta"], p["nu"], p["rho"], p["v0"])
        iv = [implied_vol_fwd(x, 1.0, k, T, 0.0) for x, k in zip(c, K)]
        rows.append(dict(T=float(T), atm_iv=iv[1], atm_skew=(iv[2] - iv[0]) / (2 * h)))
    return pd.DataFrame(rows)


def heston_short_skew_limit(p):
    """Short-maturity limit of the Heston ATM skew: rho * nu / (4 sqrt(v0))."""
    return p["rho"] * p["nu"] / (4 * np.sqrt(p["v0"]))


def fit_power_law(T, skew):
    """|skew| ~ c * T^(-alpha) by OLS of log|skew| on log T (negative skews only)."""
    T, skew = np.asarray(T, float), np.asarray(skew, float)
    m = skew < 0
    x, y = np.log(T[m]), np.log(-skew[m])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1 - resid.var() / y.var()
    return dict(alpha=-slope, c=np.exp(intercept), r2=r2, n=int(m.sum()))


def thin_chain(chain, max_per_T=40):
    """Keep at most max_per_T options per maturity (evenly spaced in strike) to speed up calibration
    and to avoid giving more weight to maturities that simply have more listed strikes."""
    parts = []
    for T, g in chain.groupby("T"):
        g = g.sort_values("K")
        if len(g) > max_per_T:
            g = g.iloc[np.unique(np.linspace(0, len(g) - 1, max_per_T).round().astype(int))]
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)
