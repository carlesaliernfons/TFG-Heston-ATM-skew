Calibration and Feller
Two notebooks. Thesis: Sections 5.3 and 5.5, and Appendices B.7 to B.10.

1. calibration_by_maturity.ipynb  -  Heston calibrated maturity by maturity
   Question: which Heston parameters fit each maturity, and how do they change with maturity?
   - Global calibration of 12 October 2026 (all maturities at once) and calibrations maturity by maturity
     with different sets of free parameters (cases A to E), standard errors, the identification problem
     (profile in kappa) and the comparison of 7, 9 and 12 October.
   - Reads the saved data in ../Data; no Internet. About 35 minutes on a laptop.
         python -m nbconvert --to notebook --execute --inplace calibration_by_maturity.ipynb
   - Reproducibility check: the global calibration of 12/10 gives kappa 9.54, theta 0.0386, nu 1.9396,
     rho -0.605, v0 0.0096 and an error of 2.08 volatility points. Kappa is the worst-identified parameter
     and its third decimal can change from one computer to another (9.542 or 9.543).

2. feller_monte_carlo.ipynb  -  the Feller condition and the numerical methods
   Question: if the Feller condition does not hold, which numerical methods work and which do not?
   - Compares seven Monte Carlo schemes (Euler with absorption, reflection or full truncation, Milstein,
     Andersen's QE, exact chi-squared and the textbook's conditional Monte Carlo) and the decomposition
     formula with the Fourier price, as the Feller ratio 2*kappa*theta/nu^2 falls.
   - Four experiments: the ratio falls by raising nu (1), more time steps (2), S&P 500 parameters (3)
     and the ratio falls by reducing kappa with nu = 0.3 (4). Conclusions in section 7.
   - Needs no data and no Internet. About 8 minutes.
         python -m nbconvert --to notebook --execute --inplace feller_monte_carlo.ipynb

Figures
  fig_parameters_vs_T.png         nu, rho, v0 and error of each maturity (cases A, B and C)
  fig_kappa_profile.png           the identification problem at a single maturity
  fig_smiles_by_maturity.png      smiles of 4 maturities: market, global Heston, Heston by maturity
  fig_stability_days.png          the same pattern on 7, 9 and 12 October
  fig_cases_D_E.png               error of each maturity with only nu or only rho free (cases C, D and E)
  fig_density_vT.png              exact density of the variance for different Feller ratios
  fig_bias_vs_feller.png          bias of each method as a function of the Feller ratio
  fig_convergence_steps.png       the bias as the number of steps increases (ratio 0.25)
  fig_feller_kappa_vs_nu.png      the same Feller ratio by raising nu (exp. 1) or reducing kappa (exp. 4)
