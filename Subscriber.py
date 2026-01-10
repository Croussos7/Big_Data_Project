"""
SUBSCRIBER + INFERENCE PUBLISHER (FULL, SMOOTHED REGIME ENGINE)

- Subscribes to bar stream
- Bootstraps recent bars (MANDATORY)
- Computes features identical to training
- Applies HMM inference
- Applies alpha-smoothed regime probabilities
- Prints FULL p_state and p_next with names
- Publishes inference JSON to Pub/Sub
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from google.cloud import pubsub_v1, storage


# ============================================================
# CONFIG
# ============================================================
PROJECT_ID = "big-data-480618"
SUBSCRIPTION_ID = "sub-model2"
INFERENCE_TOPIC_ID = "spy-regime-inference"

GCS_BUCKET = "project-bucket-cr"
GCS_MODEL_BLOB = "models/hmm_spy_5min.joblib"

BASE_DIR = Path(__file__).resolve().parents[0]
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"

KEY_PATH = BASE_DIR/"KEY.json"

REST_BASE = "https://api.twelvedata.com/time_series"
TWELVE_API_KEY = "87bd43db037d44059f94c62f5da145dd"

SYMBOL = "SPY"
INTERVAL = "5min"

ROLLING_KEEP = 220
POSTERIOR_TAIL = 50
ALPHA = 0.85

IGNORE_OLD_MESSAGES = True


# ============================================================
# SAFETY
# ============================================================
if not KEY_PATH.exists():
    raise FileNotFoundError(KEY_PATH)

BASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PUB/SUB CLIENTS
# ============================================================
subscriber = pubsub_v1.SubscriberClient.from_service_account_file(str(KEY_PATH))
publisher = pubsub_v1.PublisherClient.from_service_account_file(str(KEY_PATH))

sub_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
topic_path = publisher.topic_path(PROJECT_ID, INFERENCE_TOPIC_ID)


# ============================================================
# HELPERS
# ============================================================
def make_empty_bars_df() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="float64"),
    })


def parse_dt(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(str(x), utc=True)
    return ts.tz_convert(None)


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "datetime": parse_dt(event.get("datetime") or event.get("time")),
        "open": float(event["open"]),
        "high": float(event["high"]),
        "low": float(event["low"]),
        "close": float(event["close"]),
        "volume": float(event.get("volume", 0)),
    }


# ============================================================
# LOAD MODEL
# ============================================================
def download_model():
    client = storage.Client.from_service_account_json(str(KEY_PATH))
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(GCS_MODEL_BLOB).download_to_filename(str(LOCAL_MODEL_PATH))


if not LOCAL_MODEL_PATH.exists():
    download_model()

bundle = joblib.load(str(LOCAL_MODEL_PATH))

scaler = bundle["scaler"]
hmm = bundle["hmm"]
feature_cols = list(bundle["feature_cols"])
STATE_NAMES = bundle["state_names"]

A = np.asarray(hmm.transmat_)
K = hmm.n_components


# ============================================================
# STATIONARY INIT + SMOOTHING
# ============================================================
def stationary_distribution(A, tol=1e-10):
    p = np.ones(len(A)) / len(A)
    for _ in range(10_000):
        p2 = p @ A
        if np.linalg.norm(p2 - p) < tol:
            break
        p = p2
    return p


stationary_prob = stationary_distribution(A)
prev_prob: Optional[np.ndarray] = None


def smooth_probs(p: np.ndarray) -> np.ndarray:
    global prev_prob
    if prev_prob is None:
        prev_prob = stationary_prob.copy()

    out = ALPHA * prev_prob + (1 - ALPHA) * p
    out /= out.sum()
    prev_prob = out
    return out


# ============================================================
# FEATURES (TRAINING-COMPATIBLE)
# ============================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    eps = 1e-12

    df["log_close"] = np.log(df["close"] + eps)
    df["ret_1"] = df["log_close"].diff()
    df["ret_12"] = df["log_close"].diff(12)
    df["ret_48"] = df["log_close"].diff(48)

    df["rv_12"] = df["ret_1"].rolling(12).std(ddof=1)
    df["rv_48"] = df["ret_1"].rolling(48).std(ddof=1)
    df["rv_156"] = df["ret_1"].rolling(156).std(ddof=1)

    df["log_rv_12"] = np.log(df["rv_12"] + eps)
    df["log_rv_48"] = np.log(df["rv_48"] + eps)
    df["log_rv_156"] = np.log(df["rv_156"] + eps)

    ma12 = df["close"].rolling(12).mean()
    ma48 = df["close"].rolling(48).mean()
    df["ma_ratio_12"] = df["close"] / (ma12 + eps)
    df["ma_ratio_48"] = df["close"] / (ma48 + eps)

    df["range_1"] = (df["high"] - df["low"]) / (df["close"] + eps)

    v = df["volume"]
    df["vol_z_48"] = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=1) + eps)

    df["log_dollar_vol"] = np.log(df["close"] * v + eps)
    df["jump_score"] = np.abs(df["ret_1"]) / (df["rv_48"] + eps)

    return df


# ============================================================
# FORMATTERS
# ============================================================
def fmt_probs(p: np.ndarray) -> str:
    return " | ".join(f"{STATE_NAMES[i]}={p[i]*100:5.2f}%" for i in range(len(p)))


def probs_dict(p: np.ndarray) -> Dict[str, float]:
    return {STATE_NAMES[i]: float(p[i]) for i in range(len(p))}


# ============================================================
# INFERENCE
# ============================================================
def infer(feats: pd.DataFrame) -> Optional[dict]:
    valid = feats.dropna(subset=feature_cols)
    if len(valid) == 0:
        return None

    X = scaler.transform(valid.tail(POSTERIOR_TAIL)[feature_cols].values)
    _, post = hmm.score_samples(X)

    p_state = smooth_probs(post[-1])
    p_next = p_state @ A

    cs = int(np.argmax(p_state))
    ns = int(np.argmax(p_next))
    last = valid.iloc[-1]

    return {
        "datetime": str(last["datetime"]),
        "close": float(last["close"]),
        "current_state": cs,
        "current_state_name": STATE_NAMES[cs],
        "current_prob": float(p_state[cs]),
        "p_state": probs_dict(p_state),
        "next_state": ns,
        "next_state_name": STATE_NAMES[ns],
        "next_prob": float(p_next[ns]),
        "p_next": probs_dict(p_next),
    }


# ============================================================
# BOOTSTRAP (MANDATORY)
# ============================================================
def fetch_recent_bars():
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": ROLLING_KEEP,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }
    r = requests.get(REST_BASE, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df.sort_values("datetime")[["datetime", "open", "high", "low", "close", "volume"]]


bars = make_empty_bars_df()
latest_dt: Optional[pd.Timestamp] = None

print(f"Bootstrapping {ROLLING_KEEP} bars...")
bars = fetch_recent_bars()
latest_dt = bars["datetime"].iloc[-1]
print(f"Bootstrapped. Latest bar: {latest_dt}")


# ============================================================
# CALLBACK
# ============================================================
def callback(message):
    global bars, latest_dt

    try:
        bar = normalize_event(json.loads(message.data.decode()))
        if IGNORE_OLD_MESSAGES and bar["datetime"] <= latest_dt:
            message.ack()
            return

        bars = pd.concat([bars, pd.DataFrame([bar])])
        bars = bars.drop_duplicates("datetime").sort_values("datetime").tail(ROLLING_KEEP)
        latest_dt = bars["datetime"].iloc[-1]

        feats = compute_features(bars)
        out = infer(feats)

        if out:
            print("\n" + "=" * 90)
            print(f"{out['datetime']} | Close={out['close']:.2f}")
            print(f"Current: {out['current_state']} ({out['current_state_name']}) prob={out['current_prob']*100:.2f}%")
            print("p_state:", fmt_probs(np.array(list(out["p_state"].values()))))
            print(f"Next:    {out['next_state']} ({out['next_state_name']}) prob={out['next_prob']*100:.2f}%")
            print("p_next :", fmt_probs(np.array(list(out["p_next"].values()))))
            print("=" * 90)

            publisher.publish(topic_path, json.dumps(out).encode())

        message.ack()

    except Exception as e:
        print("Subscriber error:", e)
        message.nack()


# ============================================================
# RUN
# ============================================================
def main():
    print("Listening on:", sub_path)
    subscriber.subscribe(sub_path, callback=callback)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
