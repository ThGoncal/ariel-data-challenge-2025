"""
src/models/predict.py — Prédiction et soumission — Ariel Data Challenge 2025
=============================================================================
Remplace le stub vide du template.
Génère la soumission Kaggle finale à partir du meilleur modèle MLflow.
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.ml.metrics import build_submission_dataframe


def predict_from_best_run(
    X_kaggle:       np.ndarray,
    planet_ids:     np.ndarray,
    best_run_id:    str,
    scaler_y,
    output_path:    str = "models/submission_final.csv",
    tracking_uri:   str = "http://mlflow:5000",
) -> pd.DataFrame:
    """
    Charge le meilleur modèle depuis MLflow et génère les prédictions Kaggle.

    Parameters
    ----------
    X_kaggle     : (n_test, n_features) — features du set test normalisées
    planet_ids   : (n_test,) — identifiants planètes Kaggle
    best_run_id  : run_id MLflow du meilleur modèle
    scaler_y     : StandardScaler fitté sur y_train (pour inversion)
    output_path  : chemin de sauvegarde de la soumission CSV
    tracking_uri : URI du serveur MLflow

    Returns
    -------
    DataFrame soumission au format Kaggle
    """
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Chargement du modèle depuis MLflow run_id={best_run_id[:8]}...")
    model_uri = f"runs:/{best_run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle sklearn chargé")
        return _predict_sklearn(model, X_kaggle, planet_ids, scaler_y, output_path)
    except Exception:
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info("Modèle PyTorch chargé")
            return _predict_pytorch(model, X_kaggle, planet_ids, scaler_y, output_path)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle : {e}")


def _predict_sklearn(model, X, planet_ids, scaler_y, output_path) -> pd.DataFrame:
    """Prédiction avec un modèle sklearn (MultiTaskLassoCV)."""
    from sklearn.utils import resample

    # Bootstrap pour estimer l'incertitude
    n_bootstrap = 20
    preds = []
    for i in range(n_bootstrap):
        X_boot = resample(X, random_state=i)
        preds.append(model.predict(X))

    preds_array = np.stack(preds, axis=0)
    mean_sc = preds_array.mean(axis=0)
    std_sc  = preds_array.std(axis=0)

    mean = scaler_y.inverse_transform(mean_sc)
    std  = np.clip(std_sc * scaler_y.scale_, 1e-9, None)

    return _save_submission(planet_ids, mean, std, output_path)


def _predict_pytorch(model, X, planet_ids, scaler_y, output_path) -> pd.DataFrame:
    """Prédiction avec le MLP hétéroscédastique PyTorch."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=64, shuffle=False
    )

    all_mu, all_sigma = [], []
    with torch.no_grad():
        for (X_b,) in loader:
            mu, sigma = model(X_b.to(device))
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())

    mean = scaler_y.inverse_transform(np.vstack(all_mu))
    std  = np.clip(np.vstack(all_sigma) * scaler_y.scale_, 1e-9, None)

    return _save_submission(planet_ids, mean, std, output_path)


def _save_submission(planet_ids, mean, std, output_path) -> pd.DataFrame:
    """Construit et sauvegarde le DataFrame de soumission."""
    df_sub = build_submission_dataframe(planet_ids, mean, std)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out, index=False)

    logger.info(f"✅ Soumission sauvegardée : {out} ({len(df_sub)} planètes)")
    return df_sub


def validate_submission_format(submission_path: str, sample_path: str) -> bool:
    """
    Valide que la soumission respecte le format Kaggle.

    Compare avec sample_submission.csv fourni par l'organisation.
    """
    df_sub    = pd.read_csv(submission_path)
    df_sample = pd.read_csv(sample_path)

    checks = {
        "Nombre de colonnes": len(df_sub.columns) == len(df_sample.columns),
        "Noms de colonnes":   list(df_sub.columns) == list(df_sample.columns),
        "Valeurs positives":  (df_sub.iloc[:, 1:] >= 0).all().all(),
        "Pas de NaN":         not df_sub.isnull().any().any(),
    }

    all_ok = all(checks.values())
    for check, ok in checks.items():
        icon = "✅" if ok else "❌"
        logger.info(f"  {icon} {check}")

    return all_ok
