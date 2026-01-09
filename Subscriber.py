"""
FINAL SUBSCRIBER (full rewrite)
- Uses your explicit service-account JSON key file (NO default credentials)
- Downloads the trained model bundle from GCS if missing locally
- Validates FeatureSpec (feature contract) + feature_cols
- Computes features with an explicit function (no base import)
- Consumes Pub/Sub messages, maintains rolling OHLCV buffer
- Runs HMM inference (posterior over last POSTERIOR_TAIL rows) + next-state distribution

Your fixed settings:
- Project: big-data-480618
- Subscription: spy-bars-sub
- Bucket: project-bucket-cr
- Base dir: C:\\Users\\crous\\MSc DATA SCIENCE\\BIG DATA\\Big_Data_Project
- GCS model object: models/hmm_spy_5min.joblib
- Local model cache: <BASE_DIR>\\artifacts\\hmm_spy_5min.joblib
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from google.cloud import pubsub_v1
from google.cloud import storage


# -------------------------
# FIXED CONFIG (as requested)
# -------------------------
PROJECT_ID = "big-data-480618"
SUBSCRIPTION_ID = "spy-bars-sub"
GCS_BUCKET = "project-bucket-cr"

BASE_DIR = Path(r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\Big_Data_Project")
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"
GCS_MODEL_BLOB = "models/hmm_spy_5min.joblib"

# Your service account key file (explicit credentials — no defaults)
KEY_PATH = Path(r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\big-data-480618-a0ab1a62384c.json")

# Rolling requirements
ROLLING_KEEP = 220          # > 156 max rolling window
POSTERIOR_TAIL = 50         # posteriors over last N valid rows

# Contract enforcement
STRICT_INTERVAL = True      # if message includes "interval", enforce it
STRICT_SYMBOL = False       # set True if you ONLY publish SPY and want strict enforcement


# -------------------------
# Safety checks
# -------------------------
if not KEY_PATH.exists():
    raise FileNotFoundError(f"Service account key not found: {KEY_PATH}")

BASE_DIR.mkdir(parents=True, exist_ok=True)


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
    """
    Accepts:
      - 'YYYY-mm-dd HH:MM:SS'
      - ISO8601 e.g. '2026-01-09T15:55:00Z'
    Returns timezone-naive timestamp for consistent rolling windows.
    """
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
    """
    Normalize incoming JSON to:
      datetime, open, high, low, close, volume, symbol, interval

    Supports keys:
      datetime/time/timestamp/t
      open/high/low/close/volume OR o/h/l/c/v
      symbol/ticker
      interval/tf/timeframe
    """
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
        raise RuntimeError(
            "Model bundle missing feature_spec. "
            "Update training script to save feature_spec, retrain, and upload again."
        )

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

    # normalize types
    spec["windows"] = {k: int(v) for k, v in windows.items()}
    spec["transforms"] = {k: bool(v) for k, v in transforms.items()}
    spec["epsilon"] = eps
    spec["bar_interval"] = bar_interval
    spec.setdefault("feature_version", "unknown")

    return spec


def compute_features_from_ohlcv(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Explicit feature computation (must match training pipeline).
    Uses spec windows/transforms/epsilon.
    """
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

    # 1) log returns
    if t.get("log_returns", True):
        df["log_close"] = np.log(df["close"] + eps)
        df["ret_1"] = df["log_close"].diff()
        df["ret_12"] = df["log_close"].diff(ret_12_w)
        df["ret_48"] = df["log_close"].diff(ret_48_w)
    else:
        df["ret_1"] = df["close"].pct_change()
        df["ret_12"] = df["close"].pct_change(ret_12_w)
        df["ret_48"] = df["close"].pct_change(ret_48_w)

    # 2) realized volatility + logs
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

    # 3) moving average ratios
    ma12 = df["close"].rolling(ma_12_w).mean()
    ma48 = df["close"].rolling(ma_48_w).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + eps)
    df["ma_ratio_48"] = df["close"] / (ma48 + eps)

    # 4) intrabar range normalized
    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + eps)

    # 5) volume z-score
    v = df["volume"].astype(float)
    v_mu = v.rolling(volz_w).mean()
    v_sd = v.rolling(volz_w).std(ddof=1)
    df["vol_z_48"] = (v - v_mu) / (v_sd + eps)

    # 6) dollar volume
    df["dollar_vol"] = df["close"] * v
    if t.get("log_dollar_vol", True):
        df["log_dollar_vol"] = np.log(df["dollar_vol"] + eps)
    else:
        df["log_dollar_vol"] = df["dollar_vol"]

    # 7) jump score
    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + eps)

    return df


# -------------------------
# Load bundle + validate contract
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
}
if not required_features.issubset(set(feature_cols)):
    raise RuntimeError(
        "feature_cols in bundle are missing required features.\n"
        f"Required: {sorted(required_features)}\n"
        f"Got: {feature_cols}"
    )

print(f"Loaded model bundle (cached locally): {LOCAL_MODEL_PATH}")
print(f"GCS source: gs://{GCS_BUCKET}/{GCS_MODEL_BLOB}")
print(f"States: {K}")
print(f"FeatureSpec version: {spec.get('feature_version')} | interval={spec.get('bar_interval')}")
print(f"Expected interval: {expected_interval!r} (STRICT_INTERVAL={STRICT_INTERVAL})")
print(f"Expected symbol: {expected_symbol!r} (STRICT_SYMBOL={STRICT_SYMBOL})")
print(f"Feature cols: {feature_cols}")


# -------------------------
# Rolling buffer + inference helpers
# -------------------------
bars = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
seen_dt = set()  # dedupe by datetime64[ns]


def infer_current_and_next(feats: pd.DataFrame) -> Optional[dict]:
    """
    Computes:
      p_state = posterior over states at the most recent row (using last POSTERIOR_TAIL rows)
      p_next  = p_state @ transmat
    """
    valid = feats.dropna(subset=feature_cols)
    if len(valid) == 0:
        return None

    tail_valid = valid.tail(POSTERIOR_TAIL)
    X = tail_valid[feature_cols].values.astype(float)
    Xz = scaler.transform(X)

    _, post = hmm.score_samples(Xz)  # (T, K)
    p_state = post[-1]

    state = int(np.argmax(p_state))
    p_next = (p_state @ A)
    next_state = int(np.argmax(p_next))

    last = valid.iloc[-1]
    return {
        "datetime": str(last["datetime"]),
        "close": float(last["close"]),
        "state": state,
        "p_state": p_state.tolist(),
        "next_state": next_state,
        "p_next_state": p_next.tolist(),
    }


# -------------------------
# Pub/Sub callback
# -------------------------
def handle_message(message: pubsub_v1.subscriber.message.Message):
    global bars, seen_dt

    try:
        event = json.loads(message.data.decode("utf-8"))
        bar = normalize_event(event)

        # Enforce contract if publisher provides these fields
        if STRICT_INTERVAL and expected_interval and bar["interval"] and bar["interval"] != expected_interval:
            raise RuntimeError(f"Interval mismatch: expected {expected_interval}, got {bar['interval']}")
        if STRICT_SYMBOL and expected_symbol and bar["symbol"] and bar["symbol"] != expected_symbol:
            raise RuntimeError(f"Symbol mismatch: expected {expected_symbol}, got {bar['symbol']}")

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

        feats = compute_features_from_ohlcv(bars, spec)

        out = infer_current_and_next(feats)
        if out is None:
            print(f"{row['datetime']} | buffering... (need more bars for rolling features)")
        else:
            p_state = np.round(np.array(out["p_state"]), 3)
            p_next = np.round(np.array(out["p_next_state"]), 3)
            print(
                f"{out['datetime']} | C={out['close']:.2f} | "
                f"state={out['state']} p_state={p_state} | "
                f"next_state={out['next_state']} p_next={p_next}"
            )

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
