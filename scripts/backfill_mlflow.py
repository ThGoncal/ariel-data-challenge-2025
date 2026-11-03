"""Rejoue l'historique d'un run incrémental (state.json) dans MLflow.

Utile quand la boucle a été lancée sans --mlflow :
    uv run python scripts/backfill_mlflow.py [--state data/processed/incremental/state.json]
"""
import argparse
import json
from pathlib import Path

import mlflow

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="data/processed/incremental/state.json")
ap.add_argument("--experiment", default="ariel-incremental")
a = ap.parse_args()

st = json.loads(Path(a.state).read_text())
mlflow.set_experiment(a.experiment)
with mlflow.start_run(run_name="backfill"):
    mlflow.log_params({"n_val": len(st["val_ids"]), "batches_done": st["batches_done"],
                       "stop_reason": st["stop_reason"] or "en cours"})
    for h in st["history"]:
        if h["score"] is not None:
            mlflow.log_metric("gll_score", h["score"], step=h["n_train"])
            mlflow.log_metric("gll_score_smoothed", h["smoothed"], step=h["n_train"])
print(f"{sum(h['score'] is not None for h in st['history'])} points loggés dans MLflow")
