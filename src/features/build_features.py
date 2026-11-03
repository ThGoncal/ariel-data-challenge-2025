"""
src/features/build_features.py — Feature engineering — Ariel Data Challenge 2025
==================================================================================
Remplace le stub du template par le feature engineering complet Ariel.
Wrapper autour de src/ml/features.py qui expose une interface compatible
avec le template (fonction build_features) tout en ajoutant les fonctions
spécifiques au projet.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.ml.features import (
    build_feature_matrix,
    build_target_matrix,
    compute_luminosity,
    compute_naive_stats,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en DataFrame enrichi avec les features dérivées.

    Appelé par le template et les scripts DVC.
    Ajoute la feature L_stefan (luminosité stellaire) et supprime Rs, Ts.

    Parameters
    ----------
    df : DataFrame issu de train_with_spectra.csv ou test_with_spectra.csv

    Returns
    -------
    DataFrame enrichi (même nombre de lignes, colonnes features étendues)
    """
    logger.info(f"Feature engineering sur {len(df)} lignes")
    df = df.copy()

    # Feature physique dérivée : luminosité stellaire (Stefan-Boltzmann)
    df["L_stefan"] = compute_luminosity(df)
    logger.info("  → L_stefan calculée (remplace Rs + Ts)")

    # Suppression des colonnes redondantes
    cols_to_drop = [c for c in ["Rs", "Ts", "e"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"  → Colonnes supprimées : {cols_to_drop}")

    logger.info(f"  → DataFrame final : {df.shape[1]} colonnes")
    return df


def prepare_train_data(
    df: pd.DataFrame,
    drop_eccentricity: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Prépare X et y pour l'entraînement à partir du CSV accumulé.

    Combine build_feature_matrix et build_target_matrix en un seul appel.

    Parameters
    ----------
    df                : DataFrame train_with_spectra.csv
    drop_eccentricity : si True, exclut la colonne 'e' (= 0 pour Ariel 2025)

    Returns
    -------
    X         : ndarray (n_samples, n_features)
    y         : ndarray (n_samples, 283)
    feat_cols : liste des noms de features
    wl_cols   : liste des noms de cibles (wl_1..wl_283)
    """
    X, feat_cols = build_feature_matrix(df, drop_eccentricity=drop_eccentricity)
    y, wl_cols   = build_target_matrix(df)

    logger.info(f"Données prêtes : X={X.shape} | y={y.shape}")
    return X, y, feat_cols, wl_cols


def prepare_test_data(
    df: pd.DataFrame,
    drop_eccentricity: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Prépare X pour le set de test Kaggle (sans labels).

    Returns
    -------
    X          : ndarray (n_samples, n_features)
    feat_cols  : liste des noms de features
    planet_ids : ndarray des identifiants planètes
    """
    X, feat_cols = build_feature_matrix(df, drop_eccentricity=drop_eccentricity)
    planet_ids   = df["planet_id"].values

    logger.info(f"Test set prêt : X={X.shape} | {len(planet_ids)} planètes")
    return X, feat_cols, planet_ids


def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Retourne un résumé statistique des features pour le monitoring.

    Utilisé par Evidently pour détecter le drift entre batchs.

    Returns
    -------
    dict : statistiques par colonne (mean, std, min, max, n_missing)
    """
    rp_cols = [c for c in df.columns if c.startswith("Rp_")]
    summary = {
        "n_samples":   len(df),
        "n_features":  len(rp_cols),
        "rp_mean":     float(df[rp_cols].mean().mean()) if rp_cols else None,
        "rp_std":      float(df[rp_cols].std().mean()) if rp_cols else None,
        "n_missing":   int(df.isnull().sum().sum()),
    }
    return summary
