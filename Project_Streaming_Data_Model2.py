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
    "pyarrow",
    "hmmlearn"
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
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


REST_BASE = "https://api.twelvedata.com/time_series"


# -------------------------
# 1) Fetch OHLCV bars
# -------------------------
def fetch_twelvedata_history(symbol: str, interval: str, outputsize: int, api_key: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }
    r = requests.get(REST_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Unexpected Twelve Data response: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Twelve Data returns newest-first; sort ascending
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


# -------------------------
# 2) Feature engineering (single-ticker, streamable)
# -------------------------
def compute_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # Log returns
    df["log_close"] = np.log(df["close"])
    df["ret_1"] = df["log_close"].diff()

    # Multi-horizon returns (trend)
    df["ret_12"] = df["log_close"].diff(12)   # ~1h if 5-min bars
    df["ret_48"] = df["log_close"].diff(48)   # ~4h if 5-min bars

    # Realized volatility (std of returns)
    df["rv_12"] = df["ret_1"].rolling(12).std(ddof=1)
    df["rv_48"] = df["ret_1"].rolling(48).std(ddof=1)
    df["rv_156"] = df["ret_1"].rolling(156).std(ddof=1)  # ~2 days @ 5m

    # Stabilize volatility distributions
    df["log_rv_12"] = np.log(df["rv_12"] + 1e-12)
    df["log_rv_48"] = np.log(df["rv_48"] + 1e-12)
    df["log_rv_156"] = np.log(df["rv_156"] + 1e-12)

    # Moving averages / trend ratios
    ma12 = df["close"].rolling(12).mean()
    ma48 = df["close"].rolling(48).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + 1e-12)
    df["ma_ratio_48"] = df["close"] / (ma48 + 1e-12)

    # Intrabar range / stress
    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)

    # Volume shock: z-score vs 48-bar rolling window
    v = df["volume"].astype(float)
    v_mu = v.rolling(48).mean()
    v_sd = v.rolling(48).std(ddof=1)
    df["vol_z_48"] = (v - v_mu) / (v_sd + 1e-12)

    # Dollar volume (liquidity proxy)
    df["dollar_vol"] = df["close"] * v
    df["log_dollar_vol"] = np.log(df["dollar_vol"] + 1e-12)

    # Jump / tail proxy: move size relative to recent vol
    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + 1e-12)

    return df


def feature_cols():
    return [
        "ret_1",
        "ret_12",
        "log_rv_12",
        "log_rv_48",
        "ma_ratio_12",
        "ma_ratio_48",
        "range_1",
        "vol_z_48",
        "log_dollar_vol",
        "jump_score",
    ]


# -------------------------
# 3) Train HMM
# -------------------------
def train_hmm(feat_df: pd.DataFrame, cols: list[str], n_states: int = 3):
    # Keep only rows where all features are available
    model_df = feat_df.dropna(subset=cols).reset_index(drop=True)

    X = model_df[cols].values.astype(float)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    hmm.fit(Xz)

    # Posteriors + predicted states (for demo/printing)
    logprob, post = hmm.score_samples(Xz)  # post: (T, K)
    states = post.argmax(axis=1)

    model_df["regime"] = states

    # Add probability columns for convenience
    for k in range(n_states):
        model_df[f"p_state_{k}"] = post[:, k]

    return scaler, hmm, model_df, float(logprob)


# -------------------------
# 4) Main: fetch -> features -> train -> print examples
# -------------------------
def main():
    api_key = os.environ.get("87bd43db037d44059f94c62f5da145dd")
    if not api_key:
        raise SystemExit("Missing TWELVE_API_KEY environment variable.")

    symbol = "SPY"
    interval = "5min"
    outputsize = 5000
    n_states = 3

    print("=== OFFLINE TRAINING ===")
    print(f"Symbol: {symbol} | interval: {interval} | outputsize: {outputsize} | n_states: {n_states}")

    # (1) Fetch
    bars = fetch_twelvedata_history(symbol, interval, outputsize, api_key)
    print("\n--- Raw bars (head) ---")
    print(bars.head(5).to_string(index=False))
    print("\n--- Raw bars (tail) ---")
    print(bars.tail(5).to_string(index=False))

    # (2) Features
    feats = compute_features_from_ohlcv(bars)
    cols = feature_cols()

    print("\n--- Feature columns used ---")
    print(cols)

    print("\n--- Feature sample (head, selected cols) ---")
    print(feats[["datetime", "close", "volume"] + cols].head(15).to_string(index=False))

    # Show how many NaNs you get due to rolling windows (expected at the start)
    nan_counts = feats[cols].isna().sum().sort_values(ascending=False)
    print("\n--- NaN counts per feature (expected due to rolling windows) ---")
    print(nan_counts.to_string())

    # Simple descriptive stats for sanity check
    print("\n--- Feature summary stats (after dropping NaNs) ---")
    print(feats.dropna(subset=cols)[cols].describe().to_string())

    # (3) Train HMM
    scaler, hmm, model_df, logprob = train_hmm(feats, cols, n_states=n_states)

    print("\n=== MODEL TRAINED ===")
    print(f"Training logprob (total): {logprob:.3f}")
    print("\n--- Transition matrix (transmat_) ---")
    print(pd.DataFrame(hmm.transmat_).to_string(index=False, header=False))

    print("\n--- State means (in *scaled* feature space) ---")
    # hmm.means_ is in scaled space because we fit HMM on Xz
    means_scaled = pd.DataFrame(hmm.means_, columns=cols)
    print(means_scaled.to_string(index=True))

    print("\n--- Last 10 timestamps: regimes + probabilities ---")
    view_cols = ["datetime", "close"] + ["regime"] + [f"p_state_{k}" for k in range(n_states)]
    print(model_df[view_cols].tail(10).to_string(index=False))

    # Optional: show regime counts
    print("\n--- Regime counts ---")
    print(model_df["regime"].value_counts().sort_index().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
