from __future__ import annotations
import sys
import subprocess
import time
import threading
from typing import Callable, Any, List, Optional




# ----------------- AUTO-INSTALL DEPENDENCIES -----------------

def pip_install(packages):
    """
    Install packages into the SAME interpreter that is running this script.
    """
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


required_packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "yfinance",
    "twelvedata",
    "websocket-client",
    "pyarrow"
]

pip_install(required_packages)


from twelvedata import TDClient
import pandas as pd
import numpy as np
import pyarrow as pa

print("pandas:", pd.__version__)
print("pyarrow:", pa.__version__)

# ----------------- STREAM CLASS -----------------

class TwelveDataStream:
    def __init__(
        self,
        api_key: str,
        symbols: List[str],
        on_event: Optional[Callable[[Any], None]] = None,
        heartbeat_every: int = 10,
    ):
        self.api_key = api_key
        self.symbols = symbols
        self.on_event = on_event or self.default_on_event
        self.heartbeat_every = heartbeat_every

        self._td = TDClient(apikey=self.api_key)
        self._ws = None
        self._thread = None

        self._stop_evt = threading.Event()
        self._running = False

    def default_on_event(self, event: Any) -> None:
        print("EVENT:", event)

    def _run(self) -> None:
        self._running = True
        self._stop_evt.clear()

        try:
            self._ws = self._td.websocket(on_event=self.on_event)

            # If you see no events, try swapping order:
            # self._ws.connect()
            # self._ws.subscribe(self.symbols)
            self._ws.subscribe(self.symbols)
            self._ws.connect()

            last_hb = time.time()

            while not self._stop_evt.is_set():
                now = time.time()
                if now - last_hb >= self.heartbeat_every:
                    try:
                        self._ws.heartbeat()
                    except Exception as e:
                        print("heartbeat error:", repr(e))
                    last_hb = now

                time.sleep(0.1)

        except Exception as e:
            print("stream thread error:", repr(e))

        finally:
            try:
                if self._ws is not None:
                    try:
                        self._ws.reset()
                    except Exception:
                        pass
                    try:
                        self._ws.disconnect()
                    except Exception:
                        pass
            finally:
                self._running = False

    def start(self) -> None:
        if self._running:
            print("Stream already running.")
            return

        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()
        print(f"Started Twelve Data stream for: {', '.join(self.symbols)}")

    def stop(self, join_timeout: int = 5) -> None:
        if not self._running:
            print("Stream not running.")
            return

        print("Stop requested...")
        self._stop_evt.set()

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

        if self._running:
            print("Still running (library thread likely didn’t exit cleanly).")
        else:
            print("Stream stopped.")


# ----------------- CONFIG / USAGE -----------------

# API_KEY = "87bd43db037d44059f94c62f5da145dd"
#
#
# def handle_event(event: Any) -> None:
#     print("EVENT:", event)


# def main() -> None:
#     symbols = ["AAPL"]
#     stream = TwelveDataStream(API_KEY, symbols, on_event=handle_event, heartbeat_every=10)
#
#     stream.start()
#
#     # Keep alive so you see the streaming output in PyCharm Run console
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         stream.stop()
#     finally:
#         # Best-effort stop even if PyCharm terminates the run
#         stream.stop()
#
#
# if __name__ == "__main__":
#     main()

# ------------------ GET HISTORICAL DATA ------------------

import os
import time
import pandas as pd
import requests

API_KEY = "87bd43db037d44059f94c62f5da145dd"
BASE = "https://api.twelvedata.com/time_series"

"""
End-to-end offline training script (Twelve Data -> features -> sklearn Pipeline -> saved artifacts)

What it does:
1) Downloads historical OHLCV bars from Twelve Data for a list of tickers
2) Computes features (ret_1, vol_12, vol_48, ma_ratio_12, range_1) per ticker
3) Trains an unsupervised regime model (GaussianMixture) inside a sklearn Pipeline
4) Saves artifacts to model_artifacts/v1/
   - pipeline.joblib
   - feature_spec.json
   - metadata.json

Prereqs:
  pip install pandas numpy scikit-learn joblib requests pyarrow

Auth:
  export TWELVE_API_KEY="YOUR_KEY"   (mac/linux)
  setx TWELVE_API_KEY "YOUR_KEY"     (windows powershell/cmd; reopen terminal)
"""



import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# ----------------------------
# Twelve Data download
# ----------------------------

TD_BASE = "https://api.twelvedata.com/time_series"


def fetch_twelve_data_timeseries(
    symbol: str,
    interval: str = "5min",
    outputsize: int = 5000,
    start_date: Optional[str] = None,  # e.g. "2025-12-01 00:00:00"
    end_date: Optional[str] = None,    # e.g. "2026-01-06 00:00:00"
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch OHLCV time series for one symbol from Twelve Data.

    Notes:
    - outputsize is limited by Twelve Data plan & endpoint constraints.
    - returns newest-first from API; we sort ascending.
    """
    api_key = API_KEY
    if not api_key:
        raise RuntimeError("Missing TWELVE_API_KEY. Set it as an environment variable.")

    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        # If you want adjusted prices and your plan supports it:
        # "dp": 6,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    r = requests.get(TD_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if "values" not in payload:
        raise RuntimeError(f"Twelve Data error for {symbol}: {payload}")

    df = pd.DataFrame(payload["values"])
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    # Parse types
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
    df["ticker"] = symbol
    return df


def fetch_many(
    tickers: List[str],
    interval: str = "5min",
    outputsize: int = 5000,
    throttle_seconds: float = 0.35,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        print(f"Fetching {t}...")
        df_t = fetch_twelve_data_timeseries(
            symbol=t,
            interval=interval,
            outputsize=outputsize,
            start_date=start_date,
            end_date=end_date,
        )
        dfs.append(df_t)
        time.sleep(throttle_seconds)  # helps avoid rate-limit spikes
    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# Feature engineering
# ----------------------------

def compute_features_per_ticker(
    df: pd.DataFrame,
    vol_short: int = 12,
    vol_long: int = 48,
    ma_window: int = 12,
) -> pd.DataFrame:
    """
    Compute features per ticker on OHLCV bars.

    Required columns: ticker, datetime, high, low, close (open/volume optional)
    Output columns include:
      - ret_1: log return
      - vol_12: rolling std(ret_1) over 12 bars
      - vol_48: rolling std(ret_1) over 48 bars
      - ma_ratio_12: close / SMA(close, 12)
      - range_1: (high-low)/close
    """
    required = {"ticker", "datetime", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["ticker", "datetime"]).copy()

    # Group-by ticker so rolling windows don't bleed across tickers
    g = df.groupby("ticker", group_keys=False)

    # 1-bar log return
    df["ret_1"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))

    # Rolling volatility (std of returns)
    df[f"vol_{vol_short}"] = g["ret_1"].apply(lambda s: s.rolling(vol_short).std())
    df[f"vol_{vol_long}"] = g["ret_1"].apply(lambda s: s.rolling(vol_long).std())

    # Moving average ratio
    df[f"sma_{ma_window}"] = g["close"].apply(lambda s: s.rolling(ma_window).mean())
    df[f"ma_ratio_{ma_window}"] = df["close"] / df[f"sma_{ma_window}"]

    # Intrabar range normalized by close
    df["range_1"] = (df["high"] - df["low"]) / df["close"]

    return df


"""
Example: Fetch Magnificent 7 from Twelve Data and calculate features.

Magnificent 7 tickers (commonly):
AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA

Prereqs:
  pip install pandas numpy requests pyarrow
"""

def fetch_twelve_data_timeseries(symbol: str, interval: str = "5min", outputsize: int = 5000) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
    }
    r = requests.get(TD_BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if "values" not in payload:
        raise RuntimeError(f"Twelve Data error for {symbol}: {payload}")

    df = pd.DataFrame(payload["values"])
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
    df["ticker"] = symbol
    return df


def fetch_many(tickers, interval="5min", outputsize=5000, throttle_seconds=0.35) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        print(f"Fetching {t}...")
        dfs.append(fetch_twelve_data_timeseries(t, interval=interval, outputsize=outputsize))
        time.sleep(throttle_seconds)
    return pd.concat(dfs, ignore_index=True)


def compute_features_per_ticker(df: pd.DataFrame, vol_short=12, vol_long=48, ma_window=12) -> pd.DataFrame:
    required = {"ticker", "datetime", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["ticker", "datetime"]).copy()
    g = df.groupby("ticker", group_keys=False)

    # log return
    df["ret_1"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))

    # rolling vol of returns
    df[f"vol_{vol_short}"] = g["ret_1"].apply(lambda s: s.rolling(vol_short).std())
    df[f"vol_{vol_long}"] = g["ret_1"].apply(lambda s: s.rolling(vol_long).std())

    # moving average ratio
    df[f"sma_{ma_window}"] = g["close"].apply(lambda s: s.rolling(ma_window).mean())
    df[f"ma_ratio_{ma_window}"] = df["close"] / df[f"sma_{ma_window}"]

    # intrabar range
    df["range_1"] = (df["high"] - df["low"]) / df["close"]

    return df


if __name__ == "__main__":
    MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

    # Pull latest ~5000 bars per ticker (5m interval)
    raw = fetch_many(MAG7, interval="5min", outputsize=5000, throttle_seconds=0.4)
    print("Raw rows:", len(raw))
    print(raw.head())

    # Compute features
    feats = compute_features_per_ticker(raw, vol_short=12, vol_long=48, ma_window=12)

    feature_cols = ["ret_1", "vol_12", "vol_48", "ma_ratio_12", "range_1"]

    # Show a clean sample (drop rows that don't have full rolling windows yet)
    sample = feats.dropna(subset=feature_cols).groupby("ticker").tail(3)
    print("\nSample with features (last 3 rows per ticker):")
    print(sample[["ticker", "datetime", "close"] + feature_cols].to_string(index=False))

    # (Optional) save to parquet for training
    feats.to_parquet("mag7_5min_with_features.parquet", index=False)
    print("\nSaved: mag7_5min_with_features.parquet")
