"""
TFG - daily snapshot of S&P 500 option chains (Yahoo Finance).

Every run saves, in a folder named after the date of the quotes (the last trading session):
    Data/YYYY-MM-DD/SPX.csv.gz    full ^SPX option chain (calls and puts, all expiries up to ~2 years)
    Data/YYYY-MM-DD/SPY.csv.gz    full SPY option chain
    Data/YYYY-MM-DD/market.csv    spot of ^SPX and SPY, VIX and the 13-week T-bill yield (^IRX)
If the market is open when the script runs, the folder gets the suffix "_intraday" so that the
closing snapshot of that day is still taken in the evening.
Everything that happens is written to Data/log.txt (the script runs in the background).

Robustness (added on 14/10/2026, after a run that stopped halfway):
  - while it runs, the script asks Windows not to go to sleep;
  - VIX and ^IRX are downloaded first and market.csv is written as soon as the SPX chain is saved
    (SPX is the chain the TFG uses; SPY comes afterwards);
  - every run first repairs earlier days that stopped halfway: if a day has SPX.csv.gz but no market.csv,
    market.csv is rebuilt from Yahoo's daily history (the SPY chain of that day cannot be recovered).

Usage:  py download_daily.py            (normally launched by the Windows Task Scheduler)
        py download_daily.py --repair   (only repairs earlier days; it can be run at any time)
"""
import datetime as dt
import os
import sys
import time
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
TICKERS = {"SPX": "^SPX", "SPY": "SPY"}            # SPX first: it is the chain the TFG uses
EXTRA = {"VIX": "^VIX", "IRX_13w_tbill_pct": "^IRX"}
MAX_DAYS = 800          # expiries up to ~2 years
RETRIES = 3


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with open(os.path.join(BASE, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    """Ask Windows not to go to sleep while this script runs (the request ends when the script exits)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass  # not Windows (or not available): nothing to do


def retry(fn, what):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # network hiccups, rate limits...
            if attempt == RETRIES:
                raise
            log(f"   retry {attempt}/{RETRIES - 1} for {what}: {repr(e)[:120]}")
            time.sleep(10 * attempt)


def market_open_now():
    """True if the US equity session is open right now (Mon-Fri, 9:30-16:15 New York time)."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    return now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 15)


def last_close(yf, symbol):
    hist = retry(lambda: yf.Ticker(symbol).history(period="5d")["Close"].dropna(), f"{symbol} history")
    return float(hist.iloc[-1]), hist.index[-1].date()


def close_on(yf, symbol, day):
    """Closing value of `symbol` in the session of `day` (a past date), from Yahoo's daily history."""
    hist = retry(lambda: yf.Ticker(symbol).history(start=str(day), end=str(day + dt.timedelta(days=1)))["Close"].dropna(),
                 f"{symbol} history of {day}")
    hist = hist[[d.date() == day for d in hist.index]]
    if hist.empty:
        raise ValueError(f"no close of {symbol} on {day}")
    return float(hist.iloc[-1])


def repair_incomplete(yf, current):
    """Days before `current` with SPX.csv.gz but no market.csv (a run that stopped halfway):
    market.csv is rebuilt with the spot saved in SPX.csv.gz and the closes of that day from Yahoo's history."""
    import pandas as pd

    for name in sorted(os.listdir(BASE)):
        folder = os.path.join(BASE, name)
        try:
            day = dt.date.fromisoformat(name)       # only folders named YYYY-MM-DD (not "_intraday")
        except ValueError:
            continue
        if (day >= current or os.path.exists(os.path.join(folder, "market.csv"))
                or not os.path.exists(os.path.join(folder, "SPX.csv.gz"))):
            continue
        try:
            spx = pd.read_csv(os.path.join(folder, "SPX.csv.gz"), usecols=["spot", "downloaded_at"], nrows=1)
            market = {"asof": str(day), "downloaded_at": spx["downloaded_at"].iloc[0], "session_open": False,
                      "spot_SPX": float(spx["spot"].iloc[0]), "spot_SPY": close_on(yf, TICKERS["SPY"], day)}
            for col, symbol in EXTRA.items():
                market[col] = close_on(yf, symbol, day)
            missing = [n for n in TICKERS if not os.path.exists(os.path.join(folder, f"{n}.csv.gz"))]
            market["note"] = (f"rebuilt from daily history on {dt.date.today()}"
                              + (f"; chains not saved: {', '.join(missing)}" if missing else ""))
            pd.DataFrame([market]).to_csv(os.path.join(folder, "market.csv"), index=False)
            log(f"{day}: repaired, market.csv rebuilt from daily history (chains not saved: {', '.join(missing) or 'none'})")
        except Exception as e:
            log(f"{day}: could not repair ({repr(e)[:120]}), will try again on the next run")


def chain(yf, symbol, asof):
    tk = yf.Ticker(symbol)
    expiries = retry(lambda: list(tk.options), f"{symbol} expiries")
    frames = []
    for e in expiries:
        days = (dt.date.fromisoformat(e) - asof).days
        if days < 0 or days > MAX_DAYS:
            continue
        try:
            ch = retry(lambda: tk.option_chain(e), f"{symbol} {e}")
        except Exception as ex:
            log(f"   {symbol} {e}: skipped ({repr(ex)[:100]})")
            continue
        for kind, d in (("call", ch.calls), ("put", ch.puts)):
            d = d.copy()
            d["type"], d["expiry"], d["T"] = kind, e, days / 365.0
            frames.append(d)
        time.sleep(0.3)  # be gentle with Yahoo
    import pandas as pd

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), len(expiries)


def main(only_repair=False):
    import pandas as pd
    import yfinance as yf

    keep_awake()
    intraday = market_open_now()
    spot_spx, asof = last_close(yf, TICKERS["SPX"])
    repair_incomplete(yf, asof)
    if only_repair:
        log("repair finished")
        return
    folder = os.path.join(BASE, f"{asof}" + ("_intraday" if intraday else ""))
    done = os.path.join(folder, "market.csv")
    if os.path.exists(done) and not intraday:
        log(f"{asof}: already downloaded, nothing to do")
        return
    os.makedirs(folder, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    market = {"asof": str(asof), "downloaded_at": stamp, "session_open": intraday}

    for name, symbol in EXTRA.items():  # first VIX and ^IRX (two quick downloads)
        try:
            market[name], _ = last_close(yf, symbol)
        except Exception as e:
            log(f"   {symbol}: not available ({repr(e)[:100]})")

    saved = []
    for name, symbol in TICKERS.items():
        spot, _ = last_close(yf, symbol)
        df, n_exp = chain(yf, symbol, asof)
        if df.empty:
            log(f"{asof}: {symbol} returned no options")
            continue
        saved.append(name)
        df["spot"], df["asof"], df["downloaded_at"] = spot, str(asof), stamp
        df.to_csv(os.path.join(folder, f"{name}.csv.gz"), index=False, compression="gzip")
        market[f"spot_{name}"] = spot
        n_quotes = int(((df["bid"] > 0) & (df["ask"] > 0)).sum()) if "bid" in df else 0
        log(f"{asof}: {symbol} saved: {len(df)} options, {df['expiry'].nunique()} of {n_exp} expiries, {n_quotes} with bid/ask")
        if "SPX" in saved and all(k in market for k in EXTRA):  # the day is usable as soon as SPX is saved
            pd.DataFrame([market]).to_csv(done, index=False)

    if not os.path.exists(done):  # no SPX (or no VIX/^IRX): the next run repairs or retries this day
        log(f"{asof}: INCOMPLETE ({len(saved)}/{len(TICKERS)} chains saved), will retry on the next run")
        return
    if len(saved) < len(TICKERS):
        log(f"{asof}: SPX saved, but not {', '.join(n for n in TICKERS if n not in saved)}")
    log(f"{asof}: done -> {folder}")


if __name__ == "__main__":
    try:
        main(only_repair="--repair" in sys.argv)
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        sys.exit(1)
