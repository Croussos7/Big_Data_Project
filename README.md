# Project – Streaming Data & Regime Model

### Overview
---
This project implements an **end-to-end streaming data pipeline** for financial market data, combining **real-time ingestion**, **feature engineering**, **Hidden Markov Model (HMM) regime inference**, and **cloud-based storage and analytics**.

The system is designed to operate on streaming time-series data, perform **online regime classification**, and publish results to **Google Cloud Platform (GCP)** services for downstream analysis.


 ### Repository Contents 📁
---

The repository contains the following components:

1. **Project_Streaming_Data_Model Script**  
   Training script used to compute features and train the Hidden Markov Model (HMM) on historical data.

2. **Publisher Script**  
   Publishes streaming market data to a cloud messaging topic, enabling real-time processing.

3. **Subscriber Script**  
   Subscribes to the streaming data, performs feature computation and HMM inference, and publishes regime probabilities and next-state estimates to a GCS topic.  
   A separate subscription consumes this data and loads it directly into **BigQuery** for analysis and visualization.

4. **Save_Artifacts Script**  
   Names regimes, saves artifacts and uploads to Google Cloud

5. **Saved Artifacts**  
   Contains trained model artifacts (e.g. serialized HMM models) used during live inference. They are already uploaded in GCS the scripts pull them from there so theoretically you shouldn't use them manually.

- The **GCS Service Account Key** will be attached in the submission. It is used to authenticate and upload data and artifacts to **Google Cloud Storage (GCS)**.

---
### Considerations and Usage

#### 1. All scripts resolve paths relative to the BASE directory, defined as =>

```
BASE_DIR = Path(__file__).resolve().parents[1]

Example usage:

LOCAL_MODEL_PATH = BASE_DIR / "artifacts" / "hmm_spy_5min.joblib"
KEY_PATH = BASE_DIR / "big-data-480618-a0ab1a62384c.json"
```
All files and folders must remain in the same BASE directory. Moving files outside this structure will break pathing in the scripts.

#### 2. Environment Setup => 
Python 3.9+ is recommended. Required packages are installed via pip (see training script first few lines). The GCS Service Account Key, should be named ***"KEY.json"***
   
#### 3. Execution =>

- Do not rerun the training or the saving artifacts script. The model should be saved in the GCS bucket and pulled from the subscriber script, applying the model using the published new candles.

- *Run the Publisher* =>
Starts streaming market data (e.g. SPY).
Candle updates occur every 5 minutes when the market is open.

- *Run the Subscriber* =>
Must run in parallel with the Publisher.
The Subscriber consumes published data, performs inference, and publishes results into the inference topic.

When running locally, this may require:

Separate terminals, or,
Parallel run configurations in an IDE (e.g. PyCharm)

Have fun!
