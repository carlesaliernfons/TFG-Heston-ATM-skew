# The ATM Implied Volatility Skew under the Heston Model

Code and data of the Final Year Project (BSc in Economics, Universitat Pompeu Fabra, 2026-2027, code ECC19)
by **Carles Aliern Fons**, tutored by Professor Elisa Alòs.

*The ATM Implied Volatility Skew under the Heston Model: Decomposition-Formula Approximations versus S&P 500 Options*

> **Status: IN PROGRESS.** The thesis PDF in `Thesis/` is a working version, not the final one.
> Content and results may change before the final submission (3 Dec 2026).

## Contents

| Folder | What it contains | Thesis |
|---|---|---|
| `Data/` | Daily S&P 500 (SPX) and SPY option chains from Yahoo Finance, one folder per day (`SPX.csv.gz`, `SPY.csv.gz`, `market.csv`), and the program that downloads them (`download_daily.py`) | Section 3 |
| `Textbook and SPY/` | Replication of the textbook experiment and first calibration of Heston to SPY options | Appendix B.2 |
| `ATM Skew/` | Market ATM skew, power-law fit, Heston and decomposition-formula skews, robustness of the skew estimator | Sections 5.1, 5.2, 5.4; Appendix B.6 |
| `Calibration and Feller/` | Heston calibrated maturity by maturity (cases A-E); the Feller condition and the Monte Carlo experiments | Sections 5.3, 5.5 |
| `Exact-error calibration/` | Calibration by maturity with the exact implied-volatility error on eight days; conditional Monte Carlo against Fourier; choice of the interest rate | Sections 3.3, 4.2, 5.3, 5.5 |
| `Rho-nu and Feller ratio/` | Product rho*nu and Feller ratio by maturity; implied volatility by bisection against Brent's method | Sections 3.3, 5.3, 5.5; Appendix B.4 |
| `Stress/` | Calm (12 Oct) versus stressed market (13, 19, 20 Nov 2026): calibration by maturity, global calibration, skew and decomposition error (Python scripts, see its README) | Section 5.6 |

Each folder contains the library `heston_tfg.py` (Black-Scholes, implied volatility, Heston Fourier price,
decomposition formula, data cleaning, calibration, skew estimators) and, where needed, `heston_experiment.py`
(the textbook's pricing functions and Monte Carlo schemes), in the version used by its notebooks.

## How to reproduce the results

```
pip install -r requirements.txt
jupyter notebook
```

Open the notebook of each folder and run all cells (in `Stress/`, run the three scripts listed in its README).
The notebooks read the saved data and do not use the Internet, with one exception: the first exercise
(`Textbook and SPY/heston_calibration.ipynb`) downloads the SPY option chain of the day on which it is run;
the snapshot used in the thesis is saved in that folder (`SPY_chain_2026-10-08.csv`).
The notebooks print the versions of Python and of the packages, and the Monte Carlo experiments use fixed
random seeds, so running them again gives the same numbers.

Yahoo Finance does not keep a history of option quotes, so the files in `Data/` cannot be downloaded again
for past days.

## Use of AI

The code was written with the help of AI tools (Claude, by Anthropic) and checked by the author, as stated
in the declaration on the use of AI in the thesis.
