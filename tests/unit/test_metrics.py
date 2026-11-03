"""
test_metrics.py — Tests unitaires pour src/ml/metrics.py
=========================================================
La métrique GLL est le cœur du challenge — ses tests sont critiques.
On vérifie la correction mathématique, les cas limites, et la protection
contre les mauvais formats de soumission.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.metrics import (
    score,
    build_submission_dataframe,
    evaluate_on_validation,
    ParticipantVisibleError,
)

# =============================================================================
# HELPERS DE TEST
# =============================================================================

N_WL      = 283
N_SAMPLES = 10


def _make_solution(y_true: np.ndarray, planet_ids: np.ndarray) -> pd.DataFrame:
    """Construit un DataFrame solution (ground truth) au format attendu par score()."""
    cols = {f"wl_{i}": y_true[:, i - 1] for i in range(1, N_WL + 1)}
    df   = pd.DataFrame({"planet_id": planet_ids})
    return pd.concat([df, pd.DataFrame(cols)], axis=1)


def _make_submission(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    planet_ids: np.ndarray,
) -> pd.DataFrame:
    """Construit un DataFrame submission au format Kaggle."""
    return build_submission_dataframe(planet_ids, y_mean, y_std)


# =============================================================================
# score()
# =============================================================================

class TestScore:

    def test_prediction_parfaite_donne_score_proche_1(self):
        """
        Si on prédit exactement les vraies valeurs avec le bruit instrumental vrai,
        le score doit être très proche de 1.0 (oracle).
        """
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        # Prédiction parfaite : μ = y_true, σ = bruit instrumental
        sigma_true = np.concatenate([[1e-6], np.ones(N_WL - 1) * 1e-5])
        y_std      = np.tile(sigma_true, (N_SAMPLES, 1))

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_true, y_std, planet_ids)

        result = score(df_solution, df_submission, "planet_id",
                       naive_mean=float(np.mean(y_true)),
                       naive_sigma=float(np.std(y_true)))

        assert result > 0.99, f"Score oracle attendu ≈ 1.0, obtenu {result:.4f}"

    def test_prediction_naive_donne_score_proche_0(self):
        """
        Si on prédit toujours la moyenne globale avec l'écart-type global,
        le score doit être proche de 0.0 (baseline naïve).
        """
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        naive_mean  = float(np.mean(y_true))
        naive_sigma = float(np.std(y_true))

        # Prédiction naïve : μ = moyenne globale, σ = écart-type global
        y_mean = np.full_like(y_true, naive_mean)
        y_std  = np.full_like(y_true, naive_sigma)

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_mean, y_std, planet_ids)

        result = score(df_solution, df_submission, "planet_id",
                       naive_mean=naive_mean, naive_sigma=naive_sigma)

        assert abs(result) < 0.05, f"Score naïf attendu ≈ 0.0, obtenu {result:.4f}"

    def test_score_clippe_entre_0_et_1(self):
        """Le score doit toujours être dans [0, 1]."""
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        # Prédictions très mauvaises (loin de y_true, avec σ minuscule)
        y_mean = np.full_like(y_true, 0.5)   # très loin des vraies valeurs
        y_std  = np.full_like(y_true, 1e-15)  # σ quasi-nul

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_mean, y_std, planet_ids)

        result = score(df_solution, df_submission, "planet_id",
                       naive_mean=0.1, naive_sigma=0.01)

        assert 0.0 <= result <= 1.0, f"Score hors [0,1] : {result}"

    def test_ne_mute_pas_le_dataframe_solution(self):
        """
        CORRECTION BUG 5 : score() ne doit pas modifier le DataFrame solution.
        C'était le bug de l'original (del solution[col] mutait l'argument).
        """
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_true,
                                          np.full_like(y_true, 1e-5),
                                          planet_ids)

        cols_avant = set(df_solution.columns)
        score(df_solution, df_submission, "planet_id",
              naive_mean=0.1, naive_sigma=0.01)
        cols_apres = set(df_solution.columns)

        assert cols_avant == cols_apres, (
            f"score() a modifié le DataFrame solution ! "
            f"Colonnes supprimées : {cols_avant - cols_apres}"
        )

    def test_ne_mute_pas_le_dataframe_submission(self):
        """score() ne doit pas modifier le DataFrame submission non plus."""
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_true,
                                          np.full_like(y_true, 1e-5),
                                          planet_ids)

        cols_avant = set(df_submission.columns)
        score(df_solution, df_submission, "planet_id",
              naive_mean=0.1, naive_sigma=0.01)
        cols_apres = set(df_submission.columns)

        assert cols_avant == cols_apres

    def test_erreur_valeurs_negatives_dans_submission(self):
        """Une ParticipantVisibleError doit être levée pour des valeurs négatives."""
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        y_mean_negatif = np.full_like(y_true, -0.1)  # valeurs négatives !
        y_std          = np.full_like(y_true, 1e-5)

        df_solution   = _make_solution(y_true, planet_ids)
        df_submission = _make_submission(y_mean_negatif, y_std, planet_ids)

        with pytest.raises(ParticipantVisibleError, match="[Nn]égatives"):
            score(df_solution, df_submission, "planet_id",
                  naive_mean=0.1, naive_sigma=0.01)

    def test_erreur_mauvais_nombre_colonnes(self):
        """Une ParticipantVisibleError si le nombre de colonnes submission est faux."""
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)

        df_solution = _make_solution(y_true, planet_ids)

        # Submission avec seulement N_WL colonnes (au lieu de 2×N_WL)
        bad_submission = pd.DataFrame(
            np.ones((N_SAMPLES, N_WL)),
            columns=[f"wl_{i}" for i in range(1, N_WL + 1)]
        )
        bad_submission.insert(0, "planet_id", planet_ids)

        with pytest.raises(ParticipantVisibleError, match="[Nn]ombre de colonnes"):
            score(df_solution, bad_submission, "planet_id",
                  naive_mean=0.1, naive_sigma=0.01)

    def test_meilleure_prediction_donne_meilleur_score(self):
        """
        Un modèle meilleur doit avoir un score GLL plus élevé.
        Test de monotonicité : si erreur ↓ alors score ↑.
        """
        rng        = np.random.default_rng(42)
        y_true     = rng.uniform(0.08, 0.12, size=(N_SAMPLES, N_WL))
        planet_ids = np.arange(N_SAMPLES)
        naive_mean  = float(np.mean(y_true))
        naive_sigma = float(np.std(y_true))

        # Modèle A : erreur relative de 10%
        y_mean_A = y_true * rng.uniform(0.90, 1.10, size=y_true.shape)
        y_std_A  = np.full_like(y_true, 6e-3)  # sigma calibré sur l'erreur (~10% de 0.1)

        # Modèle B : erreur relative de 1% (meilleur)
        y_mean_B = y_true * rng.uniform(0.99, 1.01, size=y_true.shape)
        y_std_B  = np.full_like(y_true, 6e-4)  # sigma calibré sur l'erreur (~1%)

        score_A = score(_make_solution(y_true, planet_ids),
                        _make_submission(y_mean_A, y_std_A, planet_ids),
                        "planet_id", naive_mean, naive_sigma)

        score_B = score(_make_solution(y_true, planet_ids),
                        _make_submission(y_mean_B, y_std_B, planet_ids),
                        "planet_id", naive_mean, naive_sigma)

        assert score_B > score_A, (
            f"Modèle B (erreur 1%) devrait scorer mieux que A (erreur 10%), "
            f"mais score_A={score_A:.4f} ≥ score_B={score_B:.4f}"
        )


# =============================================================================
# build_submission_dataframe
# =============================================================================

class TestBuildSubmissionDataframe:

    def test_shape_correcte(self, y_pred_mean, y_pred_std, planet_ids):
        """Le DataFrame doit avoir (n_samples, 1 + 283 + 283) = 567 colonnes."""
        df = build_submission_dataframe(planet_ids, y_pred_mean, y_pred_std)
        assert df.shape == (len(planet_ids), 1 + N_WL * 2)

    def test_colonne_planet_id_presente(self, y_pred_mean, y_pred_std, planet_ids):
        """La colonne planet_id doit être présente et en première position."""
        df = build_submission_dataframe(planet_ids, y_pred_mean, y_pred_std)
        assert df.columns[0] == "planet_id"
        np.testing.assert_array_equal(df["planet_id"].values, planet_ids)

    def test_colonnes_wl_et_std_presentes(self, y_pred_mean, y_pred_std, planet_ids):
        """Les colonnes wl_1..283 et wl_1_std..283_std doivent être présentes."""
        df = build_submission_dataframe(planet_ids, y_pred_mean, y_pred_std)
        assert "wl_1" in df.columns
        assert "wl_283" in df.columns
        assert "wl_1_std" in df.columns
        assert "wl_283_std" in df.columns

    def test_valeurs_preservees(self, y_pred_mean, y_pred_std, planet_ids):
        """Les valeurs de y_pred_mean et y_pred_std doivent être dans le DataFrame."""
        df = build_submission_dataframe(planet_ids, y_pred_mean, y_pred_std)
        np.testing.assert_array_almost_equal(
            df[[f"wl_{i}" for i in range(1, N_WL + 1)]].values,
            y_pred_mean
        )
        np.testing.assert_array_almost_equal(
            df[[f"wl_{i}_std" for i in range(1, N_WL + 1)]].values,
            y_pred_std
        )


# =============================================================================
# evaluate_on_validation
# =============================================================================

class TestEvaluateOnValidation:

    def test_retourne_un_float(self, y_train_array, y_pred_mean, y_pred_std, planet_ids):
        """evaluate_on_validation doit retourner un float dans [0, 1]."""
        naive_mean  = float(np.mean(y_train_array))
        naive_sigma = float(np.std(y_train_array))

        result = evaluate_on_validation(
            y_val_true=y_train_array,
            y_pred_mean=y_pred_mean,
            y_pred_std=y_pred_std,
            planet_ids_val=planet_ids,
            naive_mean=naive_mean,
            naive_sigma=naive_sigma,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_cohérence_avec_score_direct(
        self, y_train_array, y_pred_mean, y_pred_std, planet_ids
    ):
        """
        evaluate_on_validation doit donner le même résultat que score()
        appelé directement avec les mêmes données.
        """
        naive_mean  = float(np.mean(y_train_array))
        naive_sigma = float(np.std(y_train_array))

        # Calcul via le wrapper
        result_wrapper = evaluate_on_validation(
            y_val_true=y_train_array,
            y_pred_mean=y_pred_mean,
            y_pred_std=y_pred_std,
            planet_ids_val=planet_ids,
            naive_mean=naive_mean,
            naive_sigma=naive_sigma,
        )

        # Calcul direct via score()
        wl_cols     = [f"wl_{i}" for i in range(1, N_WL + 1)]
        df_solution = pd.DataFrame(y_train_array, columns=wl_cols)
        df_solution.insert(0, "planet_id", planet_ids)
        df_submission = build_submission_dataframe(planet_ids, y_pred_mean, y_pred_std)

        result_direct = score(
            df_solution, df_submission, "planet_id", naive_mean, naive_sigma
        )

        np.testing.assert_almost_equal(result_wrapper, result_direct, decimal=10)
