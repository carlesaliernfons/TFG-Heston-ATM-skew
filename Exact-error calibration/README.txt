Exact-error calibration
Thesis: Sections 3.3, 4.2, 5.3 and 5.5.

exact_error_calibration.ipynb
  1. Global calibration of 12 October 2026 with the exact error in implied volatility (instead of the
     price error divided by the vega).
  2. Heston calibrated maturity by maturity with the exact error on eight days (7 to 19 October 2026),
     with kappa and theta fixed.
  3. Where the textbook's conditional Monte Carlo differs from the Fourier price, and why.
  4. Which interest rate is used to compute the implied volatilities.
  Reads the saved data in ../Data; no Internet. About 10 minutes.
      python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 exact_error_calibration.ipynb

Files
  heston_tfg.py, heston_experiment.py   the library and the textbook's pricing functions
  fig_parameters_by_maturity.png        nu, rho and v0 by maturity on the eight days
  fig_mc_vs_fourier.png                 conditional Monte Carlo minus Fourier, by strike and maturity
