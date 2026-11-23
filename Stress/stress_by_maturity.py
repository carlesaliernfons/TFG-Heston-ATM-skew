"""Calibration by maturity (case B: kappa and theta fixed, nu, rho and v0 free) on a calm day and three stressed days.
Writes stress_by_maturity.csv. Run from this folder: python stress_by_maturity.py (about 3 minutes)."""
import time, pandas as pd
import heston_tfg as ht
from stress_lib import load_day_clean, calibrate, KAPPA, THETA

DAYS = ["2026-10-12", "2026-11-13", "2026-11-19", "2026-11-20"]       # one calm day and three stressed days
fixed = {"kappa": KAPPA, "theta": THETA}                                # global values of 12 October (calibration_by_maturity.ipynb)
p0 = {"nu": 1.9396, "rho": -0.605, "v0": 0.0096}                         # starting point: global calibration of 12 October
starts = [p0, dict(p0, nu=3.0, rho=-0.5), dict(p0, nu=4.5, rho=-0.7)]    # three starting points
rows, t0 = [], time.time()
for day in DAYS:
    r, chain, quality = load_day_clean(day)                              # expiries with stale or one-sided quotes are dropped
    print(day, "dropped expiries (days):", quality.loc[~quality["ok"], "days"].tolist(), flush=True)
    cal = ht.thin_chain(chain, max_per_T=40)                             # at most 40 options per maturity
    for T, g in cal.groupby("T"):
        p, rmse = calibrate(g, r, fixed, starts)                         # exact error in implied volatility
        rows.append(dict(day=day, days=int(round(T * 365)), n=len(g), nu=p["nu"], rho=p["rho"], v0=p["v0"], error=rmse))
    print("  done", round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).to_csv("stress_by_maturity.csv", index=False)
