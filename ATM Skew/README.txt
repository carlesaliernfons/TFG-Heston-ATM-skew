ATM Skew
Question: do the Heston model and the decomposition formula reproduce how the ATM skew of implied
volatility changes with maturity in S&P 500 options?
Thesis: Sections 5.1, 5.2 and 5.4, and Appendix B.6.

atm_skew.ipynb
  - The whole analysis, with every line commented.
  - It reads the saved data (data/SPX_2026-10-08.csv, options at the close of 7 October 2026) and the
    interest rate from ../Data/2026-10-07/market.csv. It does not use the Internet, so it always gives
    the same result.
  - To run it: "Run All" in Jupyter or VS Code (about 2 minutes), or from a terminal in this folder:
        python -m nbconvert --to notebook --execute --inplace atm_skew.ipynb
  - Section 7 checks the robustness (width of the band, step h, ranges of maturities and bias of the
    skew estimator) and section 8 has the results, the conclusions and the references.

Files
  heston_tfg.py               the library (prices, implied volatility, cleaning, calibration, skew estimators)
  heston_experiment.py        the textbook's conditional Monte Carlo
  data/                       the saved option chain of 7 October 2026
  fig_skew_term_structure.png the main figure (market and model ATM skew against maturity)
  fig_smiles.png              market smiles of 7 October 2026
