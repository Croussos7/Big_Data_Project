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
]

pip_install(required_packages)


from twelvedata import TDClient


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

API_KEY = "87bd43db037d44059f94c62f5da145dd"


def handle_event(event: Any) -> None:
    print("EVENT:", event)


def main() -> None:
    symbols = ["AAPL"]
    stream = TwelveDataStream(API_KEY, symbols, on_event=handle_event, heartbeat_every=10)

    stream.start()

    # Keep alive so you see the streaming output in PyCharm Run console
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
    finally:
        # Best-effort stop even if PyCharm terminates the run
        stream.stop()


if __name__ == "__main__":
    main()

# ------------------ GET HISTORICAL DATA ------------------

import requests

API_KEY = "87bd43db037d44059f94c62f5da145dd"

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": "AAPL,TSLA",
    "interval": "5min",
    "outputsize": 5000,
    "apikey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()
print(data)
