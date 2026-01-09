import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib

from google.cloud import pubsub_v1

# Reuse feature engineering from your base script (single source of truth)
import Project_Streaming_Data_Model2 as base


# -------------------------
# CONFIG
# -------------------------
PROJECT_ID = "big-data-480618"
SUBSCRIPTION_ID = "spy-bars-sub"  # must exist and be attached to topic spy-bars

# Service account key JSON (needs Pub/Sub Subscriber role)
KEY_PATH = base.KEY_PATH  # ensure KEY_PATH is defined in your base script

# Twelve Data (bootstrap history)
TWELVE_API_KEY = base.TWELVE_API_KEY  # ensure this exists in base, or hardcode it here
REST_BASE = "https://api.twelvedata.com/time_series"

SYMBOL = "SPY"
INTERVAL = "5min"

# Local model artifact (use base variable if you have it; otherwise set it here)
try:
    LOCAL_MODEL_PATH = Path(base.LOCAL_MODEL_PATH)
except Exception:
    LOCAL_MODEL_PATH = Path(
        r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\Big_Data_Project\artifacts\hmm_spy_5min.joblib"
    )

# Rolling requirements
ROLLING_KEEP = 220        # must exceed max rolling window (156) comfortably
POSTERIOR_TAIL = 50       # compute posterior on last N valid rows (smoother P(state))


# -------------------------
# Bootstrap: fetch last N bars so inference works immediately
# -------------------------
def fetch_recent_bars(symbol: str, interval: str, n: int) -> pd.DataFrame:
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

    # oldest -> newest
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


# -------------------------
# Load model bundle (trained offline)
# -------------------------
if not LOCAL_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {LOCAL_MODEL_PATH}")

bundle = joblib.load(LOCAL_MODEL_PATH)
scaler = bundle["scaler"]
hmm = bundle["hmm"]
feature_cols = bundle["feature_cols"]  # use EXACT training columns

A = hmm.transmat_
K = hmm.n_components

print(f"Loaded model from: {LOCAL_MODEL_PATH}")
print("States:", K)
print("Feature cols:", feature_cols)


# -------------------------
# Initialize rolling buffer
# -------------------------
print(f"Bootstrapping {ROLLING_KEEP} bars for {SYMBOL} {INTERVAL} ...")
bars = fetch_recent_bars(SYMBOL, INTERVAL, ROLLING_KEEP)
bars = bars.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

if len(bars) == 0:
    raise RuntimeError("Bootstrap returned 0 bars; check Twelve Data response and API key.")

last_seen_dt = bars["datetime"].iloc[-1]
print(f"Bootstrapped {len(bars)} bars. Latest: {last_seen_dt}")


# -------------------------
# Inference helpers
# -------------------------
def infer_current_and_next(feats: pd.DataFrame) -> dict | None:
    """
    Computes:
      - p_state = P(S_t | x_1:t)
      - p_next  = P(S_{t+1} | x_1:t) = p_state @ A
    """
    valid = feats.dropna(subset=feature_cols)
    if len(valid) == 0:
        return None

    tail_valid = valid.tail(POSTERIOR_TAIL)
    X = tail_valid[feature_cols].values.astype(float)
    Xz = scaler.transform(X)

    # posteriors over states for each row in tail
    _, post = hmm.score_samples(Xz)   # (T, K)
    p_state = post[-1]               # current posterior

    state = int(np.argmax(p_state))
    p_next = (p_state @ A)           # next-state distribution
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
    global bars, last_seen_dt

    try:
        event = json.loads(message.data.decode("utf-8"))

        event_dt = pd.to_datetime(event["datetime"])

        # Ignore old/replayed messages (common after restarts)
        if event_dt <= last_seen_dt:
            message.ack()
            return

        row = {
            "datetime": event_dt,
            "open": float(event["open"]),
            "high": float(event["high"]),
            "low": float(event["low"]),
            "close": float(event["close"]),
            "volume": float(event["volume"]),
        }

        # Append + dedupe + keep tail
        bars = pd.concat([bars, pd.DataFrame([row])], ignore_index=True)
        bars = (
            bars.drop_duplicates(subset=["datetime"])
                .sort_values("datetime")
                .reset_index(drop=True)
        )
        bars = bars.tail(ROLLING_KEEP).reset_index(drop=True)

        last_seen_dt = bars["datetime"].iloc[-1]

        # Compute metrics using the base script function (single source of truth)
        feats = base.compute_features_from_ohlcv(bars)

        out = infer_current_and_next(feats)
        if out is None:
            print(f"{row['datetime']} | buffering... (unexpected after bootstrap)")
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
    subscriber = pubsub_v1.SubscriberClient.from_service_account_file(KEY_PATH)
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

