"""
src/models/evaluate.py — Évaluation — Ariel Data Challenge 2025
================================================================
Remplace le stub vide du template.
Expose la métrique GLL officielle du challenge via une interface simple,
et les fonctions d'évaluation utilisées par le pipeline incrémental.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.ml.metrics import evaluate_on_validation, build_submission_dataframe, score


def evaluate_predictions(
    y_true:      np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std:  np.ndarray,
    planet_ids:  np.ndarray,
    naive_mean:  float,
    naive_sigma: float,
) -> float:
    """
    Calcule le score GLL officiel sur un set de validation.

    Wrapper autour de evaluate_on_validation pour une interface simple.

    Parameters
    ----------
    y_true       : (n_samples, 283) — vraies valeurs
    y_pred_mean  : (n_samples, 283) — moyennes prédites
    y_pred_std   : (n_samples, 283) — écarts-types prédits
    planet_ids   : (n_samples,) — identifiants planètes
    naive_mean   : moyenne globale du train (baseline de référence)
    naive_sigma  : écart-type global du train

    Returns
    -------
    float : score GLL dans [0, 1]
    """
    val_score = evaluate_on_validation(
        y_val_true=y_true,
        y_pred_mean=y_pred_mean,
        y_pred_std=y_pred_std,
        planet_ids_val=planet_ids,
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
    )
    logger.info(f"Score GLL : {val_score:.6f}")
    return val_score


def compare_with_mlflow_best(
    current_score: float,
    experiment_name: str,
    metric_name: str = "val_gll_score",
    tracking_uri: str = "http://mlflow:5000",
) -> tuple[float, str | None]:
    """
    Compare le score courant avec le meilleur score enregistré dans MLflow.

    Utilisé par le pipeline incrémental pour décider de continuer ou d'arrêter.

    Parameters
    ----------
    current_score   : score GLL du batch courant
    experiment_name : nom de l'expérience MLflow
    metric_name     : nom de la métrique à comparer
    tracking_uri    : URI du serveur MLflow

    Returns
    -------
    best_score : float, meilleur score historique dans MLflow
    best_run_id: str ou None, run_id du meilleur modèle
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client     = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.info("Première expérience MLflow — pas d'historique")
            return -1.0, None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1,
        )

        if not runs:
            return -1.0, None

        best_run   = runs[0]
        best_score = best_run.data.metrics.get(metric_name, -1.0)
        best_run_id = best_run.info.run_id

        logger.info(
            f"Meilleur historique MLflow : {best_score:.6f} "
            f"(run_id={best_run_id[:8]}...)"
        )
        return best_score, best_run_id

    except Exception as e:
        logger.warning(f"Impossible de consulter MLflow : {e}")
        return -1.0, None


def generate_score_evolution_report(
    score_history: list[dict],
    output_path: str = "monitoring/evidently/reports/score_evolution.csv",
) -> pd.DataFrame:
    """
    Génère un rapport CSV de l'évolution du score GLL par batch.

    Contient pour chaque batch :
        - batch_idx    : indice du batch
        - n_planets    : nombre de planètes accumulées
        - val_gll      : score GLL sur la validation
        - delta        : amélioration vs batch précédent
        - is_best      : True si ce batch est le meilleur

    Parameters
    ----------
    score_history : liste de dicts {"batch", "score", "n_planets", "run_id"}
    output_path   : chemin de sauvegarde du rapport CSV

    Returns
    -------
    DataFrame du rapport
    """
    from pathlib import Path

    if not score_history:
        logger.warning("Historique vide — rapport non généré")
        return pd.DataFrame()

    df = pd.DataFrame(score_history)
    df = df.rename(columns={"score": "val_gll"})

    # Amélioration vs batch précédent
    df["delta"]   = df["val_gll"].diff().fillna(0)
    df["is_best"] = df["val_gll"] == df["val_gll"].max()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    logger.info(f"Rapport d'évolution du score sauvegardé : {output}")
    logger.info(
        f"  Meilleur score : {df['val_gll'].max():.6f} "
        f"(batch {df.loc[df['is_best'].idxmax(), 'batch']}, "
        f"{df.loc[df['is_best'].idxmax(), 'n_planets']} planètes)"
    )
    return df
