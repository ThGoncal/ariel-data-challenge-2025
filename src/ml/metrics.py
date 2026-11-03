"""
metrics.py — Métrique officielle Ariel Data Challenge 2025
===========================================================
Contient la fonction score() fournie par l'organisation,
avec les corrections et la documentation complète.

Métrique : Gaussian Log-Likelihood (GLL) normalisée.

Score = 0.0 → aussi bon que la baseline naïve (prédire toujours la moyenne)
Score = 1.0 → aussi bon que l'oracle (connaît le bruit instrumental exact)
"""

import logging
import numpy as np
import pandas as pd
import pandas.api.types
import scipy.stats

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTION PUBLIQUE
# =============================================================================

class ParticipantVisibleError(Exception):
    """Levée pour les erreurs de format de soumission."""
    pass


# =============================================================================
# MÉTRIQUE OFFICIELLE — CORRIGÉE
# =============================================================================

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 1.0,
) -> float:
    """
    Gaussian Log-Likelihood normalisée — métrique officielle du challenge.

    Pour chaque planète i et chaque longueur d'onde j, on modélise la
    prédiction comme une gaussienne 1D : N(μ_pred[i,j], σ_pred[i,j]).
    On mesure la log-vraisemblance de la vraie valeur sous cette gaussienne,
    normalisée entre la baseline naïve (0) et l'oracle physique (1).

        GLL_pred  = log P(y_true | N(y_pred,   σ_pred))
        GLL_true  = log P(y_true | N(y_true,   σ_instrument))  ← oracle
        GLL_mean  = log P(y_true | N(naive_mean, naive_sigma)) ← baseline naïve

        score_ij  = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
        score     = moyenne pondérée de score_ij, clippée dans [0, 1]

    Format de soumission Kaggle :
        planet_id | wl_1..wl_283 | wl_1_std..wl_283_std
        →  n_wavelengths moyennes + n_wavelengths écarts-types = 566 colonnes

    CORRECTION BUG 5 :
        L'original utilisait `del solution[col]` qui mute le DataFrame
        passé en argument. On utilise `.drop(columns=...)` qui retourne
        une copie — les DataFrames originaux sont préservés.

    Parameters
    ----------
    solution           : ground truth (n_samples, n_wavelengths + 1)
    submission         : prédictions  (n_samples, n_wavelengths*2 + 1)
    row_id_column_name : colonne identifiant (ex: "planet_id")
    naive_mean         : moyenne globale du train (via compute_naive_stats)
    naive_sigma        : écart-type global du train
    fsg_sigma_true     : bruit instrumental FGS1 (σ = 1e-6)
    airs_sigma_true    : bruit instrumental AIRS (σ = 1e-5)
    fgs_weight         : poids du canal FGS1 dans la moyenne finale

    Returns
    -------
    float dans [0.0, 1.0]
    """
    # CORRECTION BUG 5 : .drop() retourne une copie, pas de mutation in-place
    solution   = solution.drop(columns=[row_id_column_name])
    submission = submission.drop(columns=[row_id_column_name])

    # --- Vérifications de format
    if submission.min().min() < 0:
        raise ParticipantVisibleError(
            "Valeurs négatives dans la soumission — les spectres doivent être positifs."
        )
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"La colonne '{col}' doit être numérique.")

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError(
            f"Mauvais nombre de colonnes : {len(submission.columns)} trouvées, "
            f"{n_wavelengths * 2} attendues."
        )

    # --- Extraction
    y_pred     = submission.iloc[:, :n_wavelengths].values           # (n, 283)
    sigma_pred = np.clip(
        submission.iloc[:, n_wavelengths:].values, a_min=1e-15, a_max=None
    )                                                                 # (n, 283)

    # Bruit instrumental vrai : FGS1 (canal 0) puis AIRS (canaux 1..282)
    sigma_true = np.concatenate([
        [fsg_sigma_true],
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    ])                                                                # (283,)

    y_true = solution.values                                          # (n, 283)

    # --- Calcul GLL
    GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred,     scale=sigma_pred)
    GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true,     scale=sigma_true)
    GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma)

    # --- Normalisation et pondération
    ind_scores       = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)  # (n, 283)
    weights          = np.ones_like(ind_scores)
    weights[:, 0]    = fgs_weight                                       # poids FGS1

    result = float(np.clip(np.average(ind_scores, weights=weights), 0.0, 1.0))
    logger.info(f"Score GLL : {result:.6f}")
    return result


# =============================================================================
# UTILITAIRES D'ÉVALUATION LOCALE
# =============================================================================

def build_submission_dataframe(
    planet_ids: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_wavelengths: int = 283,
) -> pd.DataFrame:
    """
    Construit un DataFrame au format Kaggle depuis des arrays numpy.

    Format : planet_id | wl_1..wl_283 | wl_1_std..wl_283_std

    Parameters
    ----------
    planet_ids   : (n_samples,)          identifiants
    y_pred_mean  : (n_samples, 283)      moyennes prédites
    y_pred_std   : (n_samples, 283)      écarts-types prédits

    Returns
    -------
    DataFrame prêt pour submission.to_csv("submission.csv", index=False)
    """
    mean_cols = {f"wl_{i}":     y_pred_mean[:, i - 1] for i in range(1, n_wavelengths + 1)}
    std_cols  = {f"wl_{i}_std": y_pred_std[:, i - 1]  for i in range(1, n_wavelengths + 1)}

    df = pd.DataFrame({"planet_id": planet_ids})
    df = pd.concat([df, pd.DataFrame(mean_cols), pd.DataFrame(std_cols)], axis=1)
    logger.info(f"Soumission : {df.shape[0]} planètes × {df.shape[1]} colonnes")
    return df


def evaluate_on_validation(
    y_val_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    planet_ids_val: np.ndarray,
    naive_mean: float,
    naive_sigma: float,
) -> float:
    """
    Évalue les prédictions sur le set de validation local avec la métrique officielle.

    Wrapper qui formate les arrays en DataFrames puis appelle score().
    À utiliser pendant l'entraînement pour suivre la qualité sans soumettre à Kaggle.

    Returns
    -------
    float : score GLL dans [0, 1]
    """
    wl_cols     = [f"wl_{i}" for i in range(1, 284)]
    df_solution = pd.DataFrame(y_val_true, columns=wl_cols)
    df_solution.insert(0, "planet_id", planet_ids_val)

    df_submission = build_submission_dataframe(planet_ids_val, y_pred_mean, y_pred_std)

    return score(
        solution=df_solution,
        submission=df_submission,
        row_id_column_name="planet_id",
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
    )
