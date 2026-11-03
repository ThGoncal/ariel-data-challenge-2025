"""
features.py — Feature engineering Ariel Data Challenge 2025
============================================================
Responsabilité : transformer les DataFrames bruts (issus du processing)
en matrices X (features) et y (targets) prêtes pour l'entraînement.

Séparé de train.py pour être testable et réutilisable indépendamment.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================

SIGMA_SB = 5.670374419e-8   # Constante de Stefan-Boltzmann (W/m²/K⁴)
R_SUN    = 6.957e8           # Rayon solaire (m)
L_SUN    = 3.828e26          # Luminosité solaire (W)

N_WAVELENGTHS = 283          # FGS1 (1) + AIRS (282) = 283 canaux


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_luminosity(df: pd.DataFrame) -> pd.Series:
    """
    Calcule la luminosité stellaire normalisée L/L_sun via Stefan-Boltzmann.

    L = 4π R² σ T⁴    avec R en mètres, T en Kelvin
    L_norm = L / L_sun

    Cette feature remplace Rs et Ts séparément car elle encode leur
    interaction physique directe : c'est la luminosité qui conditionne
    le flux reçu par la planète et donc son spectre de transmission.

    Parameters
    ----------
    df : DataFrame contenant "Rs" (en R_sun) et "Ts" (en K)

    Returns
    -------
    Series : L/L_sun pour chaque ligne
    """
    R = df["Rs"].values * R_SUN
    T = df["Ts"].values
    L = 4 * np.pi * (R ** 2) * SIGMA_SB * (T ** 4)
    return pd.Series(L / L_SUN, index=df.index, name="L_stefan")


def build_feature_matrix(
    df: pd.DataFrame,
    drop_eccentricity: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Construit la matrice de features X.

    Features finales :
        Rp_1..283  : spectre mesuré par le traitement du signal (283 valeurs)
        Ms, Mp, P, sma, i : paramètres orbitaux/stellaires restants
        L_stefan   : luminosité stellaire (remplace Rs + Ts)

    CORRECTION BUG 4 : plus de pop() par index fragile.
    On construit la liste de features explicitement par nom.

    Parameters
    ----------
    df               : DataFrame issu de run_processing.py
    drop_eccentricity: si True, exclut 'e' (= 0 partout en 2025)

    Returns
    -------
    X        : ndarray (n_samples, n_features)
    feat_cols: liste ordonnée des noms de colonnes
    """
    df = df.copy()  # ne jamais modifier le DataFrame original

    # --- Feature physique dérivée
    df["L_stefan"] = compute_luminosity(df)

    # --- Liste de features construite EXPLICITEMENT par nom
    # Rs et Ts supprimées (redondantes avec L_stefan)
    # e supprimée car = 0 pour toutes les planètes Ariel 2025
    astro_cols = ["Ms", "Mp", "P", "sma", "i", "L_stefan"]
    if not drop_eccentricity and "e" in df.columns:
        astro_cols.insert(0, "e")

    rp_cols   = [f"Rp_{i}" for i in range(1, N_WAVELENGTHS + 1)]
    feat_cols = rp_cols + astro_cols

    # --- Vérification
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    X = df[feat_cols].values.astype(np.float64)
    logger.info(f"build_feature_matrix : X.shape={X.shape} | {len(feat_cols)} features")
    return X, feat_cols


def build_target_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extrait la matrice de targets y (spectres de transmission vrais).

    Returns
    -------
    y       : ndarray (n_samples, 283)
    wl_cols : liste des noms de colonnes cibles
    """
    wl_cols = [f"wl_{i}" for i in range(1, N_WAVELENGTHS + 1)]
    missing = [c for c in wl_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes cibles manquantes : {missing}")

    y = df[wl_cols].values.astype(np.float64)
    logger.info(f"build_target_matrix : y.shape={y.shape}")
    return y, wl_cols


def compute_naive_stats(y: np.ndarray) -> tuple[float, float]:
    """
    Calcule la moyenne et l'écart-type naïfs du train set.

    Ces valeurs sont requises par score() comme baseline de référence.
    Le modèle "naïf" prédit toujours la moyenne globale avec σ global.

    Returns
    -------
    naive_mean  : float
    naive_sigma : float
    """
    naive_mean  = float(np.mean(y))
    naive_sigma = float(np.std(y))
    logger.info(f"Baseline naïve : mean={naive_mean:.6f}  sigma={naive_sigma:.6f}")
    return naive_mean, naive_sigma
