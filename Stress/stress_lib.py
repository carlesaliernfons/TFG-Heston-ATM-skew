"""Helpers for the calm-versus-stressed comparison (Section 5, "A stressed market").
Same data loading and exact implied-volatility calibration as exact_error_calibration.ipynb, plus a quality check of each expiry."""
import os, time, numpy as np, pandas as pd
from scipy.optimize import least_squares
import heston_tfg as ht
DATA_FOLDER = os.path.join("..", "Data")   # the saved option chains (one folder per day)
KAPPA, THETA = 9.543, 0.0386
def load_day(day):
    # Reads the SPX options of one day and returns (r, clean chain with the implied volatilities)
    raw = pd.read_csv(os.path.join(DATA_FOLDER, day, "SPX.csv.gz"))         # one row per option
    market = pd.read_csv(os.path.join(DATA_FOLDER, day, "market.csv"))       # a single row
    spot = float(market["spot_SPX"].iloc[0])                                 # index level at the close
    r = float(market["IRX_13w_tbill_pct"].iloc[0]) / 100                     # 13-week T-bill yield (in %) -> decimal (section 5)
    raw["days"] = (raw["T"] * 365).round().astype(int)                      # days to expiry
    raw = raw[raw["days"] >= 2]                                              # out: options expiring today or tomorrow
    raw["root"] = raw["contractSymbol"].str[:-15]                            # "SPX" (AM) or "SPXW" (PM)
    has_spxw = raw.groupby("expiry")["root"].transform(lambda s: (s == "SPXW").any())
    raw = raw[(raw["root"] == "SPXW") | ~has_spxw]                           # SPXW when the expiry has it
    valid = (raw["bid"] > 0) & (raw["ask"] > raw["bid"])                     # live bid and ask
    n_valid = raw[valid].groupby("days").size()                              # number of valid quotes per expiry
    chosen = []
    for t in ht.TARGET_DAYS:                                                 # 2, 4, 7, ..., 365 days
        near = n_valid[(n_valid.index >= 0.85 * t - 1) & (n_valid.index <= 1.15 * t + 1)]   # expiries within ±15%
        if len(near) == 0:
            near = n_valid.iloc[[np.argmin(np.abs(n_valid.index - t))]]      # if none, the closest one
        chosen.append(int(near.idxmax()))                                    # the most liquid one
    raw = raw[raw["days"].isin(sorted(set(chosen)))]
    chain = ht.build_clean_chain(raw, spot, r=r)                             # forward, OTM options, implied vols, filters
    return r, chain


def iv_bisection(price, F, K, T, r, n_iter=60):
    # Black-76 implied volatility of many strikes at once: the σ is between 0.0001 and 5; we halve the interval 60 times
    low = np.full(len(K), 1e-4)                                              # lower end for each strike
    high = np.full(len(K), 5.0)                                              # upper end
    for _ in range(n_iter):
        mid = 0.5 * (low + high)                                             # midpoint
        too_high = ht.bs_call_fwd(F, K, T, mid, r) > price                   # is the Black-76 price with this σ too high?
        high = np.where(too_high, mid, high)                                 # yes: the right σ is on the left
        low = np.where(too_high, low, mid)                                   # no: it is on the right
    return 0.5 * (low + high)                                                # the centre of the final interval


def iv_errors(p, chain, r):
    # Exact error σ_model − σ_market of every option in `chain` (all its maturities)
    errors = []
    for T, g in chain.groupby("T"):                                          # one maturity at a time
        F, K = g["F"].iloc[0], g["K"].values                                 # forward and strikes
        prices = ht.heston_call_fwd(F, K, T, r, p["kappa"], p["theta"], p["nu"], p["rho"], p["v0"])   # Heston price (Fourier)
        errors.append(iv_bisection(prices, F, K, T, r) - g["iv"].values)    # implied vol of the model − market
    return np.concatenate(errors)


def calibrate(chain, r, fixed, starts):
    # Minimises Σ(σ_model − σ_market)² over the free parameters; `fixed` = {name: value} that are not touched
    free = [n for n in ht.PARAM_NAMES if n not in fixed]                    # parameters we look for
    pos = [ht.PARAM_NAMES.index(n) for n in free]                           # their position (for the bounds)
    scale = {"kappa": 1.0, "theta": 0.05, "nu": 0.3, "rho": 0.3, "v0": 0.05}   # typical size of each parameter
    full = lambda x: {**fixed, **dict(zip(free, x))}                         # fixed + free -> the 5 parameters
    best = None
    for s in starts:                                                         # several starting points (local minima exist)
        x0 = np.clip([s[n] for n in free], ht.LOWER[pos] + 1e-6, ht.UPPER[pos] - 1e-6)
        sol = least_squares(lambda x: iv_errors(full(x), chain, r), x0,
                            bounds=(ht.LOWER[pos], ht.UPPER[pos]), x_scale=[scale[n] for n in free])
        if best is None or sol.cost < best.cost:                             # we keep the best one
            best = sol
    rmse = 100 * np.sqrt(np.mean(best.fun ** 2))                             # error in volatility points
    return full(best.x), rmse


def quality_flags(chain, spot):
    # One row per maturity: does it have strikes on both sides of the money, and is its forward consistent?
    rows = []
    for T, g in chain.groupby("T"):
        k = np.log(g["K"] / g["F"])
        F = g["F"].iloc[0]
        rows.append(dict(T=T, days=int(round(T * 365)), n=len(g), kmin=k.min(), kmax=k.max(), n_left=int((k < 0).sum()), n_right=int((k > 0).sum()),
                         carry=np.log(F / spot) / T, F=F))
    q = pd.DataFrame(rows)
    from scipy.stats import theilslopes
    slope, icpt, _, _ = theilslopes(np.log(q["F"]), q["T"])                # robust line ln F = a + b T
    q["fwd_dev"] = np.log(q["F"]) - (icpt + slope * q["T"])                 # distance to the forward curve
    q["two_sided"] = (q["kmin"] < -0.01) & (q["kmax"] > 0.01) & (q["n_left"] >= 5) & (q["n_right"] >= 5)
    q["forward_ok"] = q["fwd_dev"].abs() < 0.005                            # within 0.5% of the curve
    q["ok"] = q["two_sided"] & q["forward_ok"]
    return q


def load_day_clean(day):
    r, chain = load_day(day)
    spot = float(pd.read_csv(os.path.join(DATA_FOLDER, day, "market.csv"))["spot_SPX"].iloc[0])
    q = quality_flags(chain, spot)
    good = q.loc[q["ok"], "T"].values
    return r, chain[chain["T"].isin(good)].copy(), q
