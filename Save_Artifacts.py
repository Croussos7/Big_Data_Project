import Project_Streaming_Data_Model2 as base
from twelvedata import TDClient
import pandas as pd
import numpy as np
import pyarrow as pa
import os
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib
from pathlib import Path
import joblib
from pathlib import Path
from google.cloud import storage

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[0]
LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"
KEY_PATH = BASE_DIR / "KEY.json"

# --------------------------------------------------
# Load existing model bundle
# --------------------------------------------------
bundle = joblib.load(LOCAL_MODEL_PATH)

print("Loaded model bundle with keys:", list(bundle.keys()))

# --------------------------------------------------
# Define regime names (POST-training interpretation)
# --------------------------------------------------
state_names = [
    "Bearish Liquidity Pressure",
    "High-Volatility Bullish Expansion",
    "Calm / Drift",
    "Stress / Sell off",
    "Calm Bullish Drift",
    "Transitionary / Low Volume",
]

# Safety check
assert len(state_names) == bundle["n_states"], \
    "Number of state names must match number of HMM states"

# --------------------------------------------------
# Enrich bundle with semantic labels
# --------------------------------------------------
bundle["state_names"] = state_names
bundle["version"] += 1

# --------------------------------------------------
# Save updated bundle locally
# --------------------------------------------------
joblib.dump(bundle, LOCAL_MODEL_PATH)
print(f"Updated model bundle saved -> {LOCAL_MODEL_PATH}")

# --------------------------------------------------
# Upload to Google Cloud Storage
# --------------------------------------------------
def upload_to_gcs_with_key(bucket_name: str, local_path: Path, gcs_path: str):
    client = storage.Client.from_service_account_json(KEY_PATH)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")

upload_to_gcs_with_key(
    bucket_name="project-bucket-cr",
    local_path=LOCAL_MODEL_PATH,
    gcs_path="models/hmm_spy_5min.joblib",
)
