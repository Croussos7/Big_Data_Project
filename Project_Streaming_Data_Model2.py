from __future__ import annotations
import sys
import subprocess
import time
import threading
from typing import Callable, Any, List, Optional
# ----------------- INSTALL PACKAGES -----------------

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
    "hmmlearn",
    "google-cloud-storage",
    "google-cloud-pubsub"
]

pip_install(required_packages)


from twelvedata import TDClient
import pandas as pd
import numpy as np
import pyarrow as pa

print("pandas:", pd.__version__)
print("pyarrow:", pa.__version__)


# ------------------ GET HISTORICAL DATA ------------------

import os
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
from pathlib import Path



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
# 2) Feature extraction
# -------------------------
def compute_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # 1. Log returns
    df["log_close"] = np.log(df["close"])
    df["ret_1"] = df["log_close"].diff()

    # 2. Multi-horizon returns (trend)
    df["ret_12"] = df["log_close"].diff(12)   # ~1h if 5-min bars
    df["ret_48"] = df["log_close"].diff(48)   # ~4h if 5-min bars

    # 3. Realized volatility becoming stationary by taking the logs (rolling standard deviation of 1-bar returns)
    df["rv_12"] = df["ret_1"].rolling(12).std(ddof=1)
    df["rv_48"] = df["ret_1"].rolling(48).std(ddof=1)
    df["rv_156"] = df["ret_1"].rolling(156).std(ddof=1)  # ~2 days @ 5m

    df["log_rv_12"] = np.log(df["rv_12"] + 1e-12)
    df["log_rv_48"] = np.log(df["rv_48"] + 1e-12)
    df["log_rv_156"] = np.log(df["rv_156"] + 1e-12)

    # 4. Moving averages / trend ratios
    ma12 = df["close"].rolling(12).mean()
    ma48 = df["close"].rolling(48).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + 1e-12)
    df["ma_ratio_48"] = df["close"] / (ma48 + 1e-12)

    # 5. Intrabar range / stress -- Normalized by 'close' to make it comparable accross price levels
    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)

    # 6. Volume shock: z-score vs 48-bar rolling window - How different is the volume in terms of std from the last 48 candles mean
    v = df["volume"].astype(float)
    v_mu = v.rolling(48).mean()
    v_sd = v.rolling(48).std(ddof=1)
    df["vol_z_48"] = (v - v_mu) / (v_sd + 1e-12)

    # 7. Dollar volume (liquidity proxy)
    df["dollar_vol"] = df["close"] * v
    df["log_dollar_vol"] = np.log(df["dollar_vol"] + 1e-12)

    # 8. Jump / tail proxy - move size relative to recent vol
    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + 1e-12)

    return df


def feature_cols():
    return [
        "ret_1",
        "ret_12",
        "log_rv_12",
        "log_rv_48",
        "log_rv_156",
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

    # Posteriors + predicted states
    logprob, post = hmm.score_samples(Xz)  # post: (T, K)
    states = post.argmax(axis=1)

    model_df["regime"] = states

    # Add probability columns for convenience
    for k in range(n_states):
        model_df[f"p_state_{k}"] = post[:, k]

    return scaler, hmm, model_df, float(logprob)


def save_artifacts(path: str, scaler, hmm, cols, interval: str, symbol: str, n_states: int):
    bundle = {
        "scaler": scaler,
        "hmm": hmm,
        "feature_cols": cols,
        "interval": interval,
        "symbol": symbol,
        "n_states": n_states,
        "version": 1,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved model bundle -> {path}")

# -------------------------
# 4) fetch -> compute features -> train --- Whole Pipeline with printed examples
# -------------------------
def RUN_PIPELINE():
    api_key = "87bd43db037d44059f94c62f5da145dd"
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

    # regime counts
    print("\n--- Regime counts ---")
    print(model_df["regime"].value_counts().sort_index().to_string())

    print("\nDone.")

    # Save Artifacts
    save_artifacts(
    path="artifacts/hmm_spy_5min.joblib",
    scaler=scaler,
    hmm=hmm,
    cols=cols,
    interval=interval,
    symbol=symbol,
    n_states=n_states,
                    )


RUN_PIPELINE()


# ----------------- UPLOAD INTO GOOGLE CLOUD -----------------

from pathlib import Path
from google.cloud import storage

BASE_DIR = Path(r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\Big_Data_Project")
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"

from pathlib import Path
from google.cloud import storage

KEY_PATH = r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\big-data-480618-a0ab1a62384c.json"

def upload_to_gcs_with_key(bucket_name: str, local_path: Path, gcs_path: str):
    client = storage.Client.from_service_account_json(KEY_PATH)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")

LOCAL_MODEL_PATH = Path(
    r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\Big_Data_Project\artifacts\hmm_spy_5min.joblib"
)

upload_to_gcs_with_key(
    bucket_name="project-bucket-cr",
    local_path=LOCAL_MODEL_PATH,
    gcs_path="models/hmm_spy_5min.joblib",
)
