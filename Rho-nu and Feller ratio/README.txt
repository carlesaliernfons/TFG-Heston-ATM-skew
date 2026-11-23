Rho-nu and Feller ratio
Thesis: Sections 3.3, 5.3 and 5.5, and Appendix B.4.

rho_nu_feller_ratio.ipynb
  1. The product rho*nu of the Heston model for each maturity, on the eight days of data.
  2. The Feller ratio 2*kappa*theta/nu^2 for each maturity.
  3. Implied volatility by bisection against Brent's method (scipy.optimize.brentq): time and precision.
  The parameters come from the calibration of ../Exact-error calibration/exact_error_calibration.ipynb
  (section 3). The first run repeats that calibration (about 7 minutes) and saves it to
  params_by_maturity.csv; later runs read the file. No Internet.

Files
  heston_tfg.py, heston_experiment.py   the library and the textbook's pricing functions
  params_by_maturity.csv                the 102 calibrations (8 days, maturity by maturity)
  fig_rho_nu_by_maturity.png            rho*nu by maturity on the eight days
