Daily S&P 500 option data (Section 3 of the thesis)

WHAT IS HERE
  One folder per trading day, named after the date of the closing quotes:
    YYYY-MM-DD/SPX.csv.gz    full option chain of ^SPX (calls and puts, expiries up to about 2 years)
    YYYY-MM-DD/SPY.csv.gz    full option chain of SPY
    YYYY-MM-DD/market.csv    price of ^SPX and SPY, the VIX and the 13-week T-bill yield (^IRX)
  The files are the raw data from Yahoo Finance (all of its columns). Yahoo Finance does not keep a
  history of option quotes, so these files cannot be downloaded again for past days.
  On 13 October 2026 only the SPX chain was saved (the SPY chain of that day is missing).

THE DOWNLOAD PROGRAM
  download_daily.py downloads the data above and saves them in a new folder. If it runs while the US
  market is open, the folder name ends in "_intraday", so that the closing snapshot is still taken later.
  Everything it does is written to log.txt.
    python download_daily.py            normal run
    python download_daily.py --repair   only rebuilds market.csv for earlier days that stopped halfway
  While it runs it asks Windows not to go to sleep. It first saves the SPX chain and market.csv (the data
  the thesis uses) and then the SPY chain.

SCHEDULING IT (Windows)
  configure_task.bat (which runs configure_task.ps1) creates the task "TFG - download options" in the
  Windows Task Scheduler: Monday to Friday at 23:00 Barcelona time, after the close of Wall Street. If the
  computer is off or asleep at that time, the task runs when it is turned on again.
  To remove it:  schtasks /Delete /TN "TFG - download options" /F

USING THE DATA
    raw = pd.read_csv("Data/2026-10-12/SPX.csv.gz")
    chain = build_clean_chain(raw, float(raw["spot"].iloc[0]), r=...)
  (build_clean_chain, in heston_tfg.py, keeps a single series when ^SPX has both SPX and SPXW options.)
