import json
import os
import time
import random
from datetime import datetime, timezone

import requests
from google.cloud import pubsub_v1


# -------------------------
# FIXED CONFIG (your project)
# -------------------------
PROJECT_ID = "big-data-480618"
TOPIC_ID = "spy-bars"

# Explicit service-account key file (NO default credentials)
PUBSUB_KEY = r"C:\Users\crous\MSc DATA SCIENCE\BIG DATA\big-data-480618-a0ab1a62384c.json"


TWELVE_API_KEY = "87bd43db037d44059f94c62f5da145dd"


REST_BASE = "https://api.twelvedata.com/time_series"

SYMBOL = "SPY"
INTERVAL = "5min"

# Polling: 5-min bars don't change every second; 25-35s is a good compromise
POLL_SECONDS = 30


# -------------------------
# Publisher client
# -------------------------
publisher = pubsub_v1.PublisherClient.from_service_account_file(PUBSUB_KEY)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

session = requests.Session()


def fetch_last_closed_bar(symbol: str = SYMBOL, interval: str = INTERVAL) -> dict:
    """
    Fetches the last CLOSED bar.
    We request outputsize=2 and publish the older one to avoid publishing an in-progress candle.
    Twelve Data typically returns newest-first in 'values'.
    """
    if not TWELVE_API_KEY:
        raise RuntimeError("Missing TWELVE_API_KEY environment variable.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 1,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }

    r = session.get(REST_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "values" not in data or not data["values"] or len(data["values"]) < 1:
        raise RuntimeError(f"Bad response / not enough bars: {data}")

    # values[0] is newest (possibly still forming), values[1] is last closed
    bar = data["values"][0]

    return {
        "symbol": symbol,
        "interval": interval,
        "datetime": bar["datetime"],  # keep as provider string; subscriber parses it
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": float(bar["volume"]),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def publish_bar(bar: dict) -> str:
    """
    Publishes the bar as Pub/Sub message data.
    Uses event_id attribute for subscriber-side dedupe.
    """
    event_id = f'{bar["symbol"]}:{bar["interval"]}:{bar["datetime"]}'
    payload = json.dumps(bar).encode("utf-8")

    # Publish synchronously for reliability (no accumulating callbacks)
    future = publisher.publish(
        topic_path,
        payload,
        symbol=bar["symbol"],
        interval=bar["interval"],
        event_id=event_id,
    )
    msg_id = future.result(timeout=20)
    print("Published", bar["datetime"], "| msg_id:", msg_id)
    return msg_id


def PUBLISHER():
    """
    Polls Twelve Data, publishes a new message only when the last CLOSED bar changes.
    """
    last_published_dt = None

    print("Publisher started")
    print("Topic:", topic_path)
    print("Symbol/Interval:", SYMBOL, INTERVAL)

    while True:
        try:
            bar = fetch_last_closed_bar()

            if bar["datetime"] != last_published_dt:
                publish_bar(bar)
                last_published_dt = bar["datetime"]
                print("BAR:", bar)
            else:
                print("No new closed bar yet.")

        except Exception as e:
            print("Publisher error:", repr(e))

        # small jitter prevents "thundering herd" patterns and rate spikes
        time.sleep(POLL_SECONDS + random.uniform(-3, 3))


if __name__ == "__main__":
    PUBLISHER()
