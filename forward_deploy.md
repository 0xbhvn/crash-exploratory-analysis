# Forward-deploying the model

---

## 1.  System at a Glance

```text
┌─────────────┐     cron / webhook      ┌─────────────────┐
│  DB / API   │ ──► fetch_incremental ─►│  Collector Job  │
└─────────────┘    (fetch_data.py)      │  (Python)       │
        ▲                               │ • appends rows  │
        │                               │ • detects 10×   │
        │ predicted_next_streak.json    │ • calls API     │
        │                               └────────┬────────┘
        │                                        │ HTTP POST /predict
        ▼                                        ▼
                                 ┌────────────────────────────┐
                                 │  FastAPI “next-streak”     │
                                 │  micro-service (Docker)    │
                                 │  – loads temporal_model    │
                                 │  – exposes /predict        │
                                 └────────┬───────────────────┘
                                          │ JSON
                                          ▼
                                 ┌────────────────────────────┐
                                 │  Sink(s):                  │
                                 │  • Kafka topic “streaks”    │
                                 │  • Postgres table           │
                                 │  • Slack / Telegram bot     │
                                 └────────────────────────────┘
```

---

## 2.  Package the Predictor (FastAPI + Docker)

> **Why FastAPI?** Zero-boilerplate, async-ready, Pydantic validation, auto-docs.

### 2.1  `app/api.py`

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from temporal.deploy import load_model_and_predict

MODEL_DIR = Path("/models")
MODEL_BUNDLE = load_model_and_predict  # alias

# preload at container startup
MODEL_PATH = MODEL_DIR / "temporal_model.pkl"
MODEL = None
def _lazy_load():
    global MODEL
    if MODEL is None:
        from joblib import load
        MODEL = load(MODEL_PATH)

class StreakRequest(BaseModel):
    # recent_​​streaks must be same schema as temporal.loader output
    recent_streaks: list[dict]  # rows ordered oldest→newest
    lookback: int | None = 50   # optional override

app = FastAPI(title="Crash-Streak Predictor")

@app.post("/predict")
def predict(req: StreakRequest):
    _lazy_load()
    df = pd.DataFrame(req.recent_streaks)
    if df.empty:
        raise HTTPException(400, "recent_streaks cannot be empty")

    # use helper from your codebase
    prediction = load_model_and_predict(MODEL_PATH, df.tail(req.lookback))
    return prediction
```

### 2.2  `Dockerfile`

```dockerfile
FROM python:3.10-slim

# system deps
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt fastapi uvicorn[standard]

# copy code + model
WORKDIR /app
COPY . /app
RUN mkdir /models && cp /app/output/temporal_model.pkl /models/

# expose & launch
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# build & push
docker build -t ghcr.io/<your-org>/streak-predictor:latest .
docker push ghcr.io/<your-org>/streak-predictor:latest
```

*Run it anywhere* – a small EC2, GCP Cloud Run, Fly.io, or Railway will do.  
Memory footprint < 500 MB, cold boot < 2 s.

---

## 3.  Collector Job (detect new 10×, call API)

Create a lightweight script that runs on a **30 s cron** (or gets kicked by a Postgres `LISTEN/NOTIFY` trigger if you prefer near-real-time).

```python
# collector.py
import os, time, requests, pandas as pd
from fetch_data import fetch_incremental_data
from data_processing import extract_streaks_and_multipliers

CSV = "games_live.csv"
API = os.getenv("PREDICT_URL", "http://streak-predictor:8000/predict")

def main():
    # 1. pull fresh rows
    fetch_incremental_data(output_file=CSV, multiplier_threshold=10.0)

    # 2. recompute streaks
    df_games = pd.read_csv(CSV)
    streaks = extract_streaks_and_multipliers(df_games, 10.0)

    # 3. if the last bust ≥ 10× just closed a streak,
    #    make a prediction for the *next* one
    last_streak = streaks.tail(1).iloc[0]
    if last_streak["hit_multiplier"] >= 10.0:
        payload = {
            "recent_streaks": streaks.to_dict(orient="records"),
            "lookback": 50
        }
        r = requests.post(API, json=payload, timeout=10)
        r.raise_for_status()
        prediction = r.json()
        print("Next-streak prediction:", prediction)
        # TODO: persist / alert as needed

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("collector error:", e)
        time.sleep(30)
```

Run this in its own container **inside the same Docker network** (so it can hit `streak-predictor:8000`), or as a Kubernetes CronJob.

---

## 4.  Persistence, Alerting, and Observability   *(plug & play)*

| Need                           | Drop-in                                    |
|--------------------------------|--------------------------------------------|
| Store every prediction         | Postgres table with `prediction` JSONB     |
| Trigger a bot message at 70 %+ | Call Slack/Telegram webhook in collector   |
| Stream to dashboards           | Produce to Kafka ↑ Grafana panel           |
| Health checks                  | `/docs` + `/predict` in FastAPI, plus K8s  |
| Metrics / traces               | Add [`prometheus_fastapi_instrumentator`]  |

---

## 5.  Keeping the Model Fresh

1. **Drift monitor**  
   Use `utils.rich_summary.display_output_summary()` on a daily Airflow DAG; if the 7-day live accuracy drops > 2 σ below training, emit a retrain event.

2. **Retrain pipeline**  
   - nightly *retrain* container (`python -m temporal.app --mode train …`)  
   - writes `temporal_model_YYYYMMDD.pkl`  
   - after evaluation, push to S3 + bump the Docker image tag via CI (GitHub Actions matrix: `model_sha=$(sha256sum *.pkl)` → image label).

3. **Blue-green rollout**  
   Deploy predictor `:canary` with new model, shadow traffic for an hour, compare accuracy, then promote.

---

## 6.  Minimal Quick-start (one-liner)

If you just want to slap it on a single box without Docker orchestration:

```bash
# inside your virtualenv
uvicorn temporal.deploy:setup_prediction_service --factory --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     --data '{"recent_streaks": [... 50 most recent rows ...]}'
```

---

### Must / Should / Could (MoSCoW)

| Priority | Item                                                                                   |
|----------|----------------------------------------------------------------------------------------|
| **Must** | Containerise predictor; collector loops & 10× detection; basic persistence/logging     |
| **Should** | Health checks, Prometheus metrics, Slack alerts on ≥ 75 % confidence longs           |
| **Could** | Auto-retrainer + blue-green rollout; HPA on request load; Web UI for manual what-ifs  |

---

**You now have an end-to-end forward-deployment pipeline**:  
*model → API → collector → continuous predictions → retrain loop*—all using the modules that already live in your repo. Plug it into your infra of choice and let it grind. 🚀
