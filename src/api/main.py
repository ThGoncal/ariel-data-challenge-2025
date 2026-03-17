"""FastAPI serving with Prometheus metrics."""
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
import time
import numpy as np

app = FastAPI(title="MLOps Template API", version="1.0.0")

# Prometheus metrics
PREDICTIONS = Counter("predictions_total", "Total predictions")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
ERRORS = Counter("prediction_errors_total", "Prediction errors")

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load model
model = mlflow.pyfunc.load_model("artifacts/model")


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: float
    probability: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        PREDICTIONS.inc()
        LATENCY.observe(time.time() - start)
        return PredictResponse(prediction=float(prediction))
    except Exception as e:
        ERRORS.inc()
        raise e
