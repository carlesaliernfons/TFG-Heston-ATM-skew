Stress: calm versus stressed market (Section 5.6 of the thesis, "A stressed market: 13-20 November 2026")

Compares 12 October 2026 (VIX 15.2) with 13, 19 and 20 November 2026 (VIX 19.3, 28.4 and 35.9).
Reads the saved chains in ../Data (no Internet). Run from this folder:

  python stress_by_maturity.py   -> stress_by_maturity.csv  (case B by maturity)
  python stress_global.py        -> g_global.csv, g_skew.csv, g_power.csv, g_decomp.csv
  python fig_stress.py           -> fig_stress.png

Quality check (stress_lib.quality_flags): an expiry is dropped if it has fewer than five strikes on
one side of the money or if its forward is more than 0.5% away from the forward curve of the other
expiries (stale quotes). Dropped: 19 Nov 132 and 315 days; 20 Nov 3, 5 and 131 days.
