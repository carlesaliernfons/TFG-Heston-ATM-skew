Textbook and SPY
First exercise of the thesis (Appendix B.2): replication of the numerical experiment of the textbook
chapter on the Heston model, and a first calibration of Heston to SPY options.

heston_calibration.ipynb
  0.  Replication of the textbook experiment (analytical Heston 9.2264, decomposition formula 9.2165,
      conditional Monte Carlo).
  1-3. Calibration of Heston to SPY options with expiries between 16 days and one year, model against
      market, decomposition formula and conditional Monte Carlo at the calibrated parameters.
  4.  Findings.

Data
  Section 1 downloads the SPY option chain of the day on which the notebook is run (it needs the Internet
  and the package yfinance) and saves a copy. The results shown in the notebook were obtained on
  8 October 2026 with the close of 7 October; that snapshot is SPY_chain_2026-10-08.csv. Running the
  notebook on another day downloads other quotes and gives different numbers. If the download fails, the
  notebook uses a synthetic market to test the code and says so ("SYNTHETIC").

Files
  heston_tfg.py, heston_experiment.py     the library and the textbook's pricing functions
  book_implied_vol.png                    implied volatility of the textbook example (Fourier and decomposition)
  market_smiles.png, calibration_fit.png, decomposition_vs_heston.png   figures of the SPY calibration
