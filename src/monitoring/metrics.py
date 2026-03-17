"""Prometheus metrics exposition."""
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions",
    ["model_name", "version"]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name"]
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy",
    ["model_name", "version"]
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score (0=no drift, 1=full drift)",
    ["feature"]
)


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus metrics HTTP server."""
    start_http_server(port)
