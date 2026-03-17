"""Run the full ML pipeline locally (outside Airflow)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from src.data.ingest import load_raw_data, save_processed_data
    logging.info("Step 1/4 — Data ingestion")
    # df = load_raw_data("your_data.csv")
    # save_processed_data(df, "processed.csv")

    logging.info("Step 2/4 — Feature engineering")
    # from src.features.build_features import build_features
    # df = build_features(df)

    logging.info("Step 3/4 — Model training")
    # from src.models.train import train_model

    logging.info("Step 4/4 — Drift monitoring")
    # from src.monitoring.drift import run_data_drift_report

    logging.info("✅ Pipeline complete")
