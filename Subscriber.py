"""
FINAL SUBSCRIBER (full, corrected)
- Uses explicit service-account JSON key file (NO default credentials)
- Downloads model bundle from GCS if missing locally
- Validates FeatureSpec + feature_cols
- Bootstraps OHLCV history from Twelve Data on startup (so inference works immediately)
- Ignores old/out-of-order Pub/Sub messages once bootstrapped
- Computes features explicitly (no base import)
- Loads state_names from the trained model bundle and prints them
- Prints:
  * current state index + state name + probability
  * full posterior p_state
  * next-state distribution p_next = p_state @ A + most likely next state

NOTE: This requires your TRAINING bundle to include:
  bundle["state_names"] = [...names...]
If missing, subscriber will fall back to ["state_0", ..., "state_{K-1}"].
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from google.cloud import pubsub_v1
from google.cloud import storage


# -------------------------
# FIXED CONFIG (your project)
# -------------------------
PROJECT_ID = "big-data-480618"
SUBSCRIPTION_ID = "spy-bars-sub-debug"
GCS_BUCKET = "project-bucket-cr"

BASE_DIR = Path(r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\Big_Data_Project")
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"
GCS_MODEL_BLOB = "models/hmm_spy_5min.joblib"

# Explicit credentials (NO default creds)
KEY_PATH = Path(r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\big-data-480618-a0ab1a62384c.json")

# Twelve Data bootstrap
REST_BASE = "https://api.twelvedata.com/time_series"
TWELVE_API_KEY = "87bd43db037d44059f94c62f5da145dd"

SYMBOL = "SPY"
INTERVAL = "5min"

ROLLING_KEEP = 220          # > 156 rolling window
POSTERIOR_TAIL = 50         # posterior window

STRICT_INTERVAL = True
STRICT_SYMBOL = False
IGNORE_OLD_MESSAGES = True


# -------------------------
# Safety checks
# -------------------------
if not KEY_PATH.exists():
    raise FileNotFoundError(f"Service account key not found: {KEY_PATH}")
BASE_DIR.mkdir(parents=True, exist_ok=True)

if not TWELVE_API_KEY:
    print("WARNING: TWELVE_API_KEY not set. Bootstrap will be skipped; subscriber will buffer until enough Pub/Sub bars arrive.")


# -------------------------
# Typed empty df (avoid pandas concat FutureWarning)
# -------------------------
def make_empty_bars_df() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="float64"),
    })


# -------------------------
# GCS: download model bundle (explicit key)
# -------------------------
def download_model_from_gcs_with_key(bucket_name: str, blob_path: str, local_path: Path) -> None:
    client = storage.Client.from_service_account_json(str(KEY_PATH))
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    print(f"Downloaded model: gs://{bucket_name}/{blob_path} -> {local_path}")


def load_bundle(bucket_name: str, blob_path: str, local_path: Path) -> Dict[str, Any]:
    if not local_path.exists():
        download_model_from_gcs_with_key(bucket_name, blob_path, local_path)
    return joblib.load(str(local_path))


# -------------------------
# Pub/Sub message parsing
# -------------------------
def parse_datetime_any(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(str(x), utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse datetime: {x!r}")
    return ts.tz_convert(None)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    dt = event.get("datetime") or event.get("time") or event.get("timestamp") or event.get("t")
    if dt is None:
        raise ValueError("Event missing datetime/time/timestamp")

    o = event.get("open", event.get("o"))
    h = event.get("high", event.get("h"))
    l = event.get("low", event.get("l"))
    c = event.get("close", event.get("c"))
    v = event.get("volume", event.get("v", 0))

    symbol = event.get("symbol") or event.get("ticker") or ""
    interval = event.get("interval") or event.get("tf") or event.get("timeframe") or ""

    bar = {
        "datetime": parse_datetime_any(dt),
        "open": safe_float(o),
        "high": safe_float(h),
        "low": safe_float(l),
        "close": safe_float(c),
        "volume": safe_float(v),
        "symbol": str(symbol),
        "interval": str(interval),
    }

    for k in ("open", "high", "low", "close"):
        if not np.isfinite(bar[k]):
            raise ValueError(f"Invalid {k} in event: {event}")
    if bar["high"] < bar["low"]:
        raise ValueError(f"High < Low in event: {event}")

    return bar


# -------------------------
# FeatureSpec validation + explicit feature computation
# -------------------------
def validate_feature_spec(bundle: Dict[str, Any]) -> Dict[str, Any]:
    spec = bundle.get("feature_spec")
    if not isinstance(spec, dict):
        raise RuntimeError("Model bundle missing feature_spec. Retrain & upload model again.")

    windows = dict(spec.get("windows", {}))
    transforms = dict(spec.get("transforms", {}))
    eps = float(spec.get("epsilon", 1e-12))
    bar_interval = str(spec.get("bar_interval", "")).strip()

    required_windows = ["ret_12", "ret_48", "rv_12", "rv_48", "rv_156", "ma_12", "ma_48", "vol_z"]
    missing = [k for k in required_windows if k not in windows]
    if missing:
        raise RuntimeError(f"feature_spec.windows missing required keys: {missing}")
    if eps <= 0:
        raise RuntimeError("feature_spec.epsilon must be > 0")

    transforms.setdefault("log_returns", True)
    transforms.setdefault("log_realized_vol", True)
    transforms.setdefault("log_dollar_vol", True)

    spec["windows"] = {k: int(v) for k, v in windows.items()}
    spec["transforms"] = {k: bool(v) for k, v in transforms.items()}
    spec["epsilon"] = eps
    spec["bar_interval"] = bar_interval
    spec.setdefault("feature_version", "unknown")
    return spec


def compute_features_from_ohlcv(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    eps = float(spec["epsilon"])
    w = spec["windows"]
    t = spec["transforms"]

    ret_12_w = int(w.get("ret_12", 12))
    ret_48_w = int(w.get("ret_48", 48))
    rv_12_w = int(w.get("rv_12", 12))
    rv_48_w = int(w.get("rv_48", 48))
    rv_156_w = int(w.get("rv_156", 156))
    ma_12_w = int(w.get("ma_12", 12))
    ma_48_w = int(w.get("ma_48", 48))
    volz_w = int(w.get("vol_z", 48))

    if t.get("log_returns", True):
        df["log_close"] = np.log(df["close"] + eps)
        df["ret_1"] = df["log_close"].diff()
        df["ret_12"] = df["log_close"].diff(ret_12_w)
        df["ret_48"] = df["log_close"].diff(ret_48_w)
    else:
        df["ret_1"] = df["close"].pct_change()
        df["ret_12"] = df["close"].pct_change(ret_12_w)
        df["ret_48"] = df["close"].pct_change(ret_48_w)

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

    ma12 = df["close"].rolling(ma_12_w).mean()
    ma48 = df["close"].rolling(ma_48_w).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + eps)
    df["ma_ratio_48"] = df["close"] / (ma48 + eps)

    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + eps)

    v = df["volume"].astype(float)
    v_mu = v.rolling(volz_w).mean()
    v_sd = v.rolling(volz_w).std(ddof=1)
    df["vol_z_48"] = (v - v_mu) / (v_sd + eps)

    df["dollar_vol"] = df["close"] * v
    df["log_dollar_vol"] = np.log(df["dollar_vol"] + eps) if t.get("log_dollar_vol", True) else df["dollar_vol"]

    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + eps)
    return df


# -------------------------
# Bootstrap history
# -------------------------
def fetch_recent_bars(symbol: str, interval: str, n: int) -> pd.DataFrame:
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY not set; cannot bootstrap.")
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": n,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }
    r = requests.get(REST_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "values" not in data or not data["values"]:
        raise RuntimeError(f"Bad response: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


# -------------------------
# Load model + validate contract + load state names
# -------------------------
bundle = load_bundle(GCS_BUCKET, GCS_MODEL_BLOB, LOCAL_MODEL_PATH)

scaler = bundle["scaler"]
hmm = bundle["hmm"]
feature_cols = list(bundle["feature_cols"])

A = np.array(hmm.transmat_, dtype=float)
K = int(hmm.n_components)

spec = validate_feature_spec(bundle)

expected_interval = str(bundle.get("interval", "")).strip() or str(spec.get("bar_interval", "")).strip()
expected_symbol = str(bundle.get("symbol", "")).strip()

required_features = {
    "ret_1", "ret_12",
    "log_rv_12", "log_rv_48", "log_rv_156",
    "ma_ratio_12", "ma_ratio_48",
    "range_1", "vol_z_48", "log_dollar_vol", "jump_score",
}
if not required_features.issubset(set(feature_cols)):
    raise RuntimeError(f"feature_cols missing required features. Got: {feature_cols}")

STATE_NAMES = bundle.get("state_names")
if not STATE_NAMES:
    print("WARNING: state_names not found in model bundle. Using numeric labels.")
    STATE_NAMES = [f"state_{i}" for i in range(K)]
if len(STATE_NAMES) != K:
    raise RuntimeError(f"state_names length ({len(STATE_NAMES)}) != K ({K})")

print(f"Loaded model: gs://{GCS_BUCKET}/{GCS_MODEL_BLOB}")
print(f"Local cache: {LOCAL_MODEL_PATH}")
print(f"K={K} | Expected interval={expected_interval!r} | Strict interval={STRICT_INTERVAL}")
print("State names:")
for i, n in enumerate(STATE_NAMES):
    print(f"  {i}: {n}")


# -------------------------
# Rolling buffer (bootstrap + dedupe)
# -------------------------
bars = make_empty_bars_df()
seen_dt = set()
latest_dt: Optional[pd.Timestamp] = None

if TWELVE_API_KEY:
    try:
        print(f"Bootstrapping {ROLLING_KEEP} bars for {SYMBOL} {INTERVAL} ...")
        bars = fetch_recent_bars(SYMBOL, INTERVAL, ROLLING_KEEP)
        seen_dt = set(pd.to_datetime(bars["datetime"]).to_numpy())
        latest_dt = bars["datetime"].iloc[-1] if len(bars) else None
        print(f"Bootstrapped {len(bars)} bars. Latest: {latest_dt}")
    except Exception as e:
        print("Bootstrap failed:", repr(e))
        bars = make_empty_bars_df()
        seen_dt = set()
        latest_dt = None


# -------------------------
# Inference + formatting
# -------------------------
def fmt_probs(vec: np.ndarray, names: list[str]) -> str:
    return " | ".join([f"{names[i]}={vec[i]*100:5.2f}%" for i in range(len(vec))])


def infer_current_and_next(feats: pd.DataFrame) -> Optional[dict]:
    valid = feats.dropna(subset=feature_cols)
    if len(valid) == 0:
        return None

    tail_valid = valid.tail(POSTERIOR_TAIL)
    X = tail_valid[feature_cols].values.astype(float)
    Xz = scaler.transform(X)

    _, post = hmm.score_samples(Xz)
    p_state = post[-1]

    current_state = int(np.argmax(p_state))
    current_prob = float(p_state[current_state])

    p_next = (p_state @ A)
    next_state = int(np.argmax(p_next))
    next_prob = float(p_next[next_state])

    last = valid.iloc[-1]
    return {
        "datetime": str(last["datetime"]),
        "close": float(last["close"]),
        "p_state": p_state,
        "current_state": current_state,
        "current_prob": current_prob,
        "p_next": p_next,
        "next_state": next_state,
        "next_prob": next_prob,
    }


# -------------------------
# Pub/Sub callback
# -------------------------
def handle_message(message: pubsub_v1.subscriber.message.Message):
    global bars, seen_dt, latest_dt

    try:
        event = json.loads(message.data.decode("utf-8"))
        bar = normalize_event(event)

        # Contract checks
        if STRICT_INTERVAL and expected_interval and bar["interval"] and bar["interval"] != expected_interval:
            message.ack()
            return
        if STRICT_SYMBOL and expected_symbol and bar["symbol"] and bar["symbol"] != expected_symbol:
            message.ack()
            return

        # Ignore old/out-of-order
        if IGNORE_OLD_MESSAGES and latest_dt is not None and bar["datetime"] <= latest_dt:
            message.ack()
            return

        dt64 = pd.to_datetime(bar["datetime"]).to_datetime64()
        if dt64 in seen_dt:
            message.ack()
            return
        seen_dt.add(dt64)

        row = {
            "datetime": bar["datetime"],
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
        }

        bars = pd.concat([bars, pd.DataFrame([row])], ignore_index=True)
        bars = (
            bars.drop_duplicates(subset=["datetime"])
                .sort_values("datetime")
                .tail(ROLLING_KEEP)
                .reset_index(drop=True)
        )
        latest_dt = bars["datetime"].iloc[-1]

        feats = compute_features_from_ohlcv(bars, spec)
        out = infer_current_and_next(feats)

        if out is None:
            print(f"{row['datetime']} | buffering... (need more bars for rolling features)")
        else:
            cs = out["current_state"]
            ns = out["next_state"]

            print("\n" + "=" * 90)
            print(f"{out['datetime']} | Close={out['close']:.2f}")
            print(f"Current: {cs} ({STATE_NAMES[cs]})  prob={out['current_prob']*100:.2f}%")
            print("p_state:", fmt_probs(out["p_state"], STATE_NAMES))
            print(f"Next:    {ns} ({STATE_NAMES[ns]})  prob={out['next_prob']*100:.2f}%")
            print("p_next :", fmt_probs(out["p_next"], STATE_NAMES))
            print("=" * 90)

        message.ack()

    except Exception as e:
        print("Subscriber error:", repr(e))
        message.nack()


# -------------------------
# Main
# -------------------------
def SUBSCRIBER():
    subscriber = pubsub_v1.SubscriberClient.from_service_account_file(str(KEY_PATH))
    sub_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    print("Listening on subscription:", sub_path)
    future = subscriber.subscribe(sub_path, callback=handle_message)

    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        print("Stopped.")


if __name__ == "__main__":
    SUBSCRIBER()
