from __future__ import annotations
import sys
import subprocess
import time
import threading
from typing import Callable, Any, List, Optional
# ----------------- INSTALL PACKAGES -----------------

import sys
import subprocess
import importlib.util

def ensure_packages(packages: dict[str, str]):
    """
    Ensures required packages are installed in the current interpreter.

    packages = {
        "import_name": "pip-install-name",
        ...
    }
    """
    missing = []

    for import_name, pip_name in packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)

    if missing:
        print("Installing missing packages:", missing)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )
    else:
        print("All required packages already installed.")

ensure_packages({
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "hmmlearn": "hmmlearn",
    "requests": "requests",
    "joblib": "joblib",
    "google.cloud.storage": "google-cloud-storage",
    "google.cloud.pubsub_v1": "google-cloud-pubsub",
    "twelvedata": "twelvedata",
})



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
FEATURE_SPEC = {
    "feature_version": "v1",
    "bar_interval": "5min",
    "windows": {
        "ret_short": 1,
        "ret_12": 12,
        "ret_48": 48,
        "rv_12": 12,
        "rv_48": 48,
        "rv_156": 156,
        "ma_12": 12,
        "ma_48": 48,
        "vol_z": 48,
    },
    "transforms": {
        "log_returns": True,
        "log_realized_vol": True,
        "log_dollar_vol": True,
    },
    "epsilon": 1e-12,
}



def compute_features_from_ohlcv_spec(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Computes features using the stored FeatureSpec.
    This ensures training and streaming inference always match exactly.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    eps = float(spec["epsilon"])
    w = spec["windows"]
    t = spec["transforms"]

    ret_12_w = int(w["ret_12"])
    ret_48_w = int(w["ret_48"])
    rv_12_w = int(w["rv_12"])
    rv_48_w = int(w["rv_48"])
    rv_156_w = int(w["rv_156"])
    ma_12_w = int(w["ma_12"])
    ma_48_w = int(w["ma_48"])
    volz_w = int(w["vol_z"])

    # 1) Returns
    if t.get("log_returns", True):
        df["log_close"] = np.log(df["close"] + eps)
        df["ret_1"] = df["log_close"].diff()
        df["ret_12"] = df["log_close"].diff(ret_12_w)
        df["ret_48"] = df["log_close"].diff(ret_48_w)
    else:
        df["ret_1"] = df["close"].pct_change()
        df["ret_12"] = df["close"].pct_change(ret_12_w)
        df["ret_48"] = df["close"].pct_change(ret_48_w)

    # 2) Realized volatility
    df["rv_12"] = df["ret_1"].rolling(rv_12_w).std(ddof=1)
    df["rv_48"] = df["ret_1"].rolling(rv_48_w).std(ddof=1)
    df["rv_156"] = df["ret_1"].rolling(rv_156_w).std(ddof=1)

    if t.get("log_realized_vol", True):
        df["log_rv_12"] = np.log(df["rv_12"] + eps)
        df["log_rv_48"] = np.log(df["rv_48"] + eps)
        df["log_rv_156"] = np.log(df["rv_156"] + eps)
    else:
        df["log_rv_12"] = df["rv_12"]
        df["log_rv_48"] = df["rv_48"]
        df["log_rv_156"] = df["rv_156"]

    # 3) Moving average ratios
    ma12 = df["close"].rolling(ma_12_w).mean()
    ma48 = df["close"].rolling(ma_48_w).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + eps)
    df["ma_ratio_48"] = df["close"] / (ma48 + eps)

    # 4) Range
    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + eps)

    # 5) Volume z-score
    v = df["volume"].astype(float)
    v_mu = v.rolling(volz_w).mean()
    v_sd = v.rolling(volz_w).std(ddof=1)
    df["vol_z_48"] = (v - v_mu) / (v_sd + eps)

    # 6) Dollar volume
    df["dollar_vol"] = df["close"] * v
    if t.get("log_dollar_vol", True):
        df["log_dollar_vol"] = np.log(df["dollar_vol"] + eps)
    else:
        df["log_dollar_vol"] = df["dollar_vol"]

    # 7) Jump score
    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + eps)

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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


def hmm_model_selection(
    feat_df: pd.DataFrame,
    cols: list[str],
    k_list: list[int],
    covariance_type: str = "full",
    n_iter: int = 500,
    random_state: int = 42,
):
    """
    Trains Gaussian HMMs for different numbers of states and compares them
    using LogLik, AIC, and BIC.

    Returns a DataFrame sorted by BIC (lower is better).
    """
    # Prepare data
    model_df = feat_df.dropna(subset=cols).reset_index(drop=True)
    X = model_df[cols].values.astype(float)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    T, D = Xz.shape
    results = []

    for K in k_list:
        hmm = GaussianHMM(
            n_components=K,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        hmm.fit(Xz)

        loglik = hmm.score(Xz)

        # ---- parameter count ----
        if covariance_type == "full":
            cov_params = D * (D + 1) // 2
        else:
            cov_params = D

        n_params = (
            (K - 1) +                 # initial probs
            K * (K - 1) +             # transition matrix
            K * D +                   # means
            K * cov_params            # covariances
        )

        aic = -2 * loglik + 2 * n_params
        bic = -2 * loglik + np.log(T) * n_params

        results.append({
            "K": K,
            "loglik": loglik,
            "AIC": aic,
            "BIC": bic,
            "n_params": n_params,
        })

    res_df = pd.DataFrame(results).sort_values("BIC").reset_index(drop=True)
    return res_df





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



def save_artifacts(path, scaler, hmm, cols, interval, symbol, n_states):
    bundle = {
        "scaler": scaler,
        "hmm": hmm,
        "feature_cols": cols,
        "interval": interval,
        "symbol": symbol,
        "n_states": n_states,
        "feature_spec": FEATURE_SPEC,
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


    # (1) Fetch
    bars = fetch_twelvedata_history(symbol, interval, outputsize, api_key)
    print("\n--- Raw bars (head) ---")
    print(bars.head(5).to_string(index=False))
    print("\n--- Raw bars (tail) ---")
    print(bars.tail(5).to_string(index=False))

    # (2) Features
    feats = compute_features_from_ohlcv_spec(bars, FEATURE_SPEC)
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
    k_candidates = [2, 3, 4, 5, 6]

    selection_df = hmm_model_selection(
        feat_df=feats,
        cols=cols,
        k_list=k_candidates,
    )

    print("\n--- HMM model selection ---")
    print(selection_df.to_string(index=False))

    best_k = int(selection_df.iloc[0]["K"])
    n_states = best_k
    print(f"\nChosen n_states by min BIC: {n_states}")

    print("=== OFFLINE TRAINING ===")
    print(f"Symbol: {symbol} | interval: {interval} | outputsize: {outputsize} | n_states: {n_states}")

    scaler, hmm, model_df, logprob = train_hmm(feats, cols, n_states=n_states)





    # -------------------------------------------------
    # Print training info
    # -------------------------------------------------
    print("\n=== MODEL TRAINED ===")
    print(f"Training logprob (total): {logprob:.3f}")



    print("\n--- State means (in *scaled* feature space) ---")
    # hmm.means_ is in scaled space because we fit HMM on Xz
    means_scaled = pd.DataFrame(hmm.means_, columns=cols)
    print(means_scaled.to_string(index=True))



    state_names = [
        "Calm / Drift",
        "Stress / Risk-Off",
        "Low-Volatility Jumps / Event-Driven",
        "Orderly Bull Trend",
        "High-Volatility Bull / Momentum",
        "Volatility without direction"
    ]

    # Safety check
    assert len(state_names) == hmm.n_components, \
        "Number of state names must match hmm.n_components"

    # -------------------------------------------------
    # explicit index → regime mapping
    # -------------------------------------------------
    print("\nState definitions:")
    for i, name in enumerate(state_names):
        print(f"State {i}: {name}")

    # 1) Build a labeled transition matrix DataFrame
    transmat_df = pd.DataFrame(
        hmm.transmat_,
        index=state_names,  # rows = FROM
        columns=state_names  # cols = TO
    )

    print("\n--- Transition matrix (P(S_t -> S_{t+1})) ---")
    print("Rows = FROM state | Columns = TO state")


    print("\n--- Transition matrix (%) ---")
    print((transmat_df * 100).round(2).to_string())

    # 3) Optional: show most likely next state for each state
    next_state = transmat_df.idxmax(axis=1)
    next_prob = transmat_df.max(axis=1)
    print("\n--- Most likely next state from each state ---")
    for s in state_names:
        print(f"{s:30s} -> {next_state[s]:30s}  p={next_prob[s]:.3f}")

    # 4) Optional: self-persistence (diagonal) ranking
    persistence = pd.Series(
        {state_names[i]: float(hmm.transmat_[i, i]) for i in range(hmm.n_components)}
    ).sort_values(ascending=False)

    print("\n--- State persistence P(stay) (diagonal) ---")
    print(persistence.to_string())

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
