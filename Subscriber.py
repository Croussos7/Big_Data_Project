
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
INFERENCE_TOPIC_ID = "spy-regime-inference-v2"

GCS_BUCKET = "project-bucket-cr"
GCS_MODEL_BLOB = "models/hmm_spy_5min.joblib"

BASE_DIR = Path(__file__).resolve().parents[0]
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"
KEY_PATH = BASE_DIR / "KEY.json"

REST_BASE = "https://api.twelvedata.com/time_series"
TWELVE_API_KEY = "87bd43db037d44059f94c62f5da145dd"

SYMBOL = "SPY"
INTERVAL = "5min"

ROLLING_KEEP = 220
IGNORE_OLD_MESSAGES = True

# RETURN HEURISTIC SETTINGS
RETURN_ALPHA = 0.7          # weight of HMM posterior
RETURN_STRENGTH = 0.4       # max heuristic strength
RETURN_SCALE = 0.002        # ~0.2% move saturates effect

NOISE_STD = 0.005            # tiny noise (visual symmetry breaker)

# Regime mapping (adjust if needed)
BULLISH_STATES = [1, 4]
BEARISH_STATES = [0, 3]


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


# ============================================================
# FORMATTERS
# ============================================================
def fmt_probs(p: np.ndarray) -> str:
    return " | ".join(
        f"{STATE_NAMES[i]}={p[i]*100:5.2f}%"
        for i in range(len(p))
    )


# ============================================================
# RETURN-BASED HEURISTIC (MAGNITUDE AWARE)
# ============================================================
def return_bias_vector(
    ret_1: float,
    K: int,
    strength: float,
) -> np.ndarray:
    bias = np.ones(K)

    if ret_1 > 0:
        for s in BULLISH_STATES:
            bias[s] += strength
        for s in BEARISH_STATES:
            bias[s] -= strength
    elif ret_1 < 0:
        for s in BEARISH_STATES:
            bias[s] += strength
        for s in BULLISH_STATES:
            bias[s] -= strength

    bias = np.clip(bias, 0.1, None)
    return bias / bias.sum()


def mix_with_return_and_noise(
    p_state: np.ndarray,
    ret_1: float,
) -> np.ndarray:
    # ---- Fix 1: magnitude-scaled strength
    scale = min(abs(ret_1) / RETURN_SCALE, 1.0)
    effective_strength = RETURN_STRENGTH * scale

    bias = return_bias_vector(
        ret_1=ret_1,
        K=len(p_state),
        strength=effective_strength,
    )

    p = RETURN_ALPHA * p_state + (1 - RETURN_ALPHA) * bias

    # ---- Fix 2: tiny noise injection
    noise = np.random.normal(0.0, NOISE_STD, size=len(p))
    p = np.clip(p + noise, 1e-6, None)

    return p / p.sum()


# ============================================================
# FEATURE COMPUTATION (LAST TIMESTAMP ONLY)
# ============================================================
def compute_last_features(df: pd.DataFrame) -> Optional[pd.Series]:
    if len(df) < 156:
        return None

    df = df.sort_values("datetime").reset_index(drop=True)
    eps = 1e-12
    i = len(df) - 1

    log_close = np.log(df["close"] + eps)
    ret_1 = log_close.diff()

    row = {}

    row["ret_1"] = ret_1.iloc[i]
    row["ret_12"] = log_close.iloc[i] - log_close.iloc[i - 12]
    row["ret_48"] = log_close.iloc[i] - log_close.iloc[i - 48]

    rv_12 = ret_1.iloc[i-11:i+1].std(ddof=1)
    rv_48 = ret_1.iloc[i-47:i+1].std(ddof=1)
    rv_156 = ret_1.iloc[i-155:i+1].std(ddof=1)

    row["log_rv_12"] = np.log(rv_12 + eps)
    row["log_rv_48"] = np.log(rv_48 + eps)
    row["log_rv_156"] = np.log(rv_156 + eps)

    ma12 = df["close"].iloc[i-11:i+1].mean()
    ma48 = df["close"].iloc[i-47:i+1].mean()

    row["ma_ratio_12"] = df["close"].iloc[i] / (ma12 + eps)
    row["ma_ratio_48"] = df["close"].iloc[i] / (ma48 + eps)

    row["range_1"] = (df["high"].iloc[i] - df["low"].iloc[i]) / (df["close"].iloc[i] + eps)

    v = df["volume"]
    v_mean = v.iloc[i-47:i+1].mean()
    v_std = v.iloc[i-47:i+1].std(ddof=1)

    row["vol_z_48"] = (v.iloc[i] - v_mean) / (v_std + eps)
    row["log_dollar_vol"] = np.log(df["close"].iloc[i] * v.iloc[i] + eps)
    row["jump_score"] = abs(ret_1.iloc[i]) / (rv_48 + eps)

    return pd.Series(row)


# ============================================================
# INFERENCE
# ============================================================
def infer_last_bar(bars: pd.DataFrame) -> Optional[dict]:
    feat = compute_last_features(bars)
    if feat is None:
        return None

    X = scaler.transform(feat[feature_cols].values.reshape(1, -1))
    _, post = hmm.score_samples(X)

    p_state = mix_with_return_and_noise(
        p_state=post[0],
        ret_1=feat["ret_1"],
    )

    p_next = p_state @ A

    cs = int(np.argmax(p_state))
    ns = int(np.argmax(p_next))
    last = bars.iloc[-1]

    return {
        "datetime": str(last["datetime"]),
        "close": float(last["close"]),
        "current_state": cs,
        "current_state_name": STATE_NAMES[cs],
        "current_prob": float(p_state[cs]),
        "p_state": {STATE_NAMES[i]: float(p_state[i]) for i in range(len(p_state))},
        "next_state": ns,
        "next_state_name": STATE_NAMES[ns],
        "next_prob": float(p_next[ns]),
        "p_next": {STATE_NAMES[i]: float(p_next[i]) for i in range(len(p_next))},
    }


# ============================================================
# BOOTSTRAP
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


bars = fetch_recent_bars()
latest_dt = bars["datetime"].iloc[-1]
print(f"Bootstrapped {len(bars)} bars. Latest: {latest_dt}")


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

        out = infer_last_bar(bars)

        if out:
            print("\n" + "=" * 90)
            print(f"{out['datetime']} | Close={out['close']:.2f}")

            p_state = np.array(list(out["p_state"].values()))
            p_next = np.array(list(out["p_next"].values()))

            print(
                f"Current: {out['current_state']} "
                f"({out['current_state_name']}) "
                f"prob={out['current_prob']*100:.2f}%"
            )
            print("p_state:", fmt_probs(p_state))

            print(
                f"Next:    {out['next_state']} "
                f"({out['next_state_name']}) "
                f"prob={out['next_prob']*100:.2f}%"
            )
            print("p_next :", fmt_probs(p_next))
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
