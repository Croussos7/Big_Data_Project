import json
import time
import requests
from datetime import datetime, timezone
from google.cloud import pubsub_v1

# Import your shared feature function from the original file
import Project_Streaming_Data_Model2 as base

PROJECT_ID = "big-data-480618"
TOPIC_ID = "spy-bars"

PUBSUB_KEY = base.KEY_PATH
TWELVE_API_KEY = "87bd43db037d44059f94c62f5da145dd"
REST_BASE = "https://api.twelvedata.com/time_series"

publisher = pubsub_v1.PublisherClient.from_service_account_file(PUBSUB_KEY)

topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)



def fetch_latest_bar(symbol="SPY", interval="5min"):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 1,
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }
    r = requests.get(REST_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "values" not in data or not data["values"]:
        raise RuntimeError(f"Bad response: {data}")

    bar = data["values"][0]  # newest
    # normalize
    return {
        "symbol": symbol,
        "interval": interval,
        "datetime": bar["datetime"],  # string
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": float(bar["volume"]),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

def publish_bar(bar: dict):
    event_id = f'{bar["symbol"]}:{bar["interval"]}:{bar["datetime"]}'
    payload = json.dumps(bar).encode("utf-8")

    future = publisher.publish(
        topic_path,
        payload,
        symbol=bar["symbol"],
        interval=bar["interval"],
        event_id=event_id,
    )

    future.add_done_callback(
        lambda f: print("Published bar:", bar["datetime"], "msg_id:", f.result())
    )

def PUBLISHER():
    last_dt = None
    while True:
        try:
            bar = fetch_latest_bar()
            if bar["datetime"] != last_dt:
                publish_bar(bar)
                last_dt = bar["datetime"]
                print("BAR:", bar)

            else:
                print("No new bar yet.")
        except Exception as e:
            print("Publisher error:", repr(e))
        time.sleep(60)

if __name__ == "__main__":
    PUBLISHER()

