"""
ariel_incremental.py — DAG Airflow : apprentissage incrémental Ariel (un run = un batch)
=========================================================================================
    run_batch ──► decide ──► trigger_next ──► (nouveau run de ce même DAG)
                    └──────► done           (arrêt : patience, max_batches, données épuisées)

Un run traite UN batch (téléchargement -> traitement -> suppression du brut -> entraînement ->
score) via IncrementalRunner.step(). L'état (state.json) vit sur disque : chaque run reprend où
le précédent s'est arrêté, et un run en échec peut être relancé sans dupliquer de données.
Pas de DAG cyclique : la boucle est obtenue par auto-déclenchement.

Lancement : Airflow UI > ariel_incremental > Trigger DAG w/ config
            {"batch_size": 20, "val_size": 40, "max_batches": 50, "patience": 3}
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DAG_ID = "ariel_incremental"


def _cfg(conf: dict):
    from src.pipelines.incremental import IncrementalConfig
    allowed = {"batch_size", "val_size", "max_batches", "patience", "min_delta",
               "smooth_window", "min_train_for_fit", "seed", "work_dir", "binning_airs", "binning_fgs1"}
    return IncrementalConfig(**{k: v for k, v in (conf or {}).items() if k in allowed})


def run_batch(**ctx) -> bool:
    """Traite un batch ; renvoie True s'il faut continuer (valeur poussée en XCom)."""
    import pandas as pd
    from src.pipelines.incremental import (IncrementalRunner, baseline_fit_predict,
                                           make_kaggle_fetch, make_model_saver, mlflow_logger)
    conf = ctx["dag_run"].conf or {}
    stars = pd.read_csv("data/raw/train_star_info.csv")
    cfg, use_ml = _cfg(conf), conf.get("mlflow", True)
    runner = IncrementalRunner(cfg, stars["planet_id"].tolist(),
                               make_kaggle_fetch(binning_airs=cfg.binning_airs, binning_fgs1=cfg.binning_fgs1),
                               baseline_fit_predict, mlflow_logger() if use_ml else None,
                               make_model_saver(cfg, use_mlflow=use_ml))
    return runner.step()


def decide(**ctx) -> str:
    return "trigger_next" if ctx["ti"].xcom_pull(task_ids="run_batch") else "done"


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2026, 1, 1),
    schedule=None,                      # Airflow >= 2.4 ; déclenché à la main puis par lui-même
    catchup=False,
    max_active_runs=1,                  # jamais deux batchs en parallèle (disque, état partagé)
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["ariel", "incremental"],
) as dag:
    t_run = PythonOperator(task_id="run_batch", python_callable=run_batch,
                           execution_timeout=timedelta(hours=3))
    t_decide = BranchPythonOperator(task_id="decide", python_callable=decide)
    t_next = TriggerDagRunOperator(task_id="trigger_next", trigger_dag_id=DAG_ID,
                                   conf="{{ dag_run.conf | tojson }}")
    t_done = EmptyOperator(task_id="done")
    t_run >> t_decide >> [t_next, t_done]
