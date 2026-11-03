"""
test_features.py — Tests unitaires pour src/ml/features.py
===========================================================
On teste le feature engineering avec des DataFrames synthétiques
dont les valeurs physiques sont connues.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import (
    compute_luminosity,
    build_feature_matrix,
    build_target_matrix,
    compute_naive_stats,
    SIGMA_SB, R_SUN, L_SUN, N_WAVELENGTHS,
)


# =============================================================================
# compute_luminosity
# =============================================================================

class TestComputeLuminosity:

    def test_soleil_donne_luminosite_1(self):
        """
        Pour Rs=1 R_sun et Ts=T_sun, la luminosité doit être exactement 1 L_sun.
        C'est la définition même des constantes — c'est le test de cohérence
        physique fondamental.
        """
        T_SUN = 5772.0   # K (température effective du Soleil)
        df    = pd.DataFrame({"Rs": [1.0], "Ts": [T_SUN]})
        result = compute_luminosity(df)

        # Tolérance de 1% pour les arrondis sur les constantes physiques
        assert abs(result.iloc[0] - 1.0) < 0.01, (
            f"L/L_sun pour le Soleil = {result.iloc[0]:.4f}, attendu 1.0"
        )

    def test_luminosite_proportionnelle_r_carre(self):
        """
        À T constant, doubler R doit quadrupler la luminosité (L ∝ R²).
        """
        T   = 5000.0
        df1 = pd.DataFrame({"Rs": [1.0], "Ts": [T]})
        df2 = pd.DataFrame({"Rs": [2.0], "Ts": [T]})

        L1 = compute_luminosity(df1).iloc[0]
        L2 = compute_luminosity(df2).iloc[0]

        np.testing.assert_almost_equal(L2 / L1, 4.0, decimal=6)

    def test_luminosite_proportionnelle_t_puissance_4(self):
        """
        À R constant, doubler T doit multiplier L par 16 (L ∝ T⁴).
        """
        R   = 1.0
        df1 = pd.DataFrame({"Rs": [R], "Ts": [3000.0]})
        df2 = pd.DataFrame({"Rs": [R], "Ts": [6000.0]})  # T × 2

        L1 = compute_luminosity(df1).iloc[0]
        L2 = compute_luminosity(df2).iloc[0]

        np.testing.assert_almost_equal(L2 / L1, 16.0, decimal=4)

    def test_toujours_positive(self, df_train_minimal):
        """La luminosité doit toujours être strictement positive."""
        result = compute_luminosity(df_train_minimal)
        assert (result > 0).all(), "Luminosité négative ou nulle détectée"

    def test_retourne_serie_pandas(self, df_train_minimal):
        """Le résultat doit être une pd.Series (pas un array numpy)."""
        result = compute_luminosity(df_train_minimal)
        assert isinstance(result, pd.Series)

    def test_index_preserve(self, df_train_minimal):
        """L'index de la Series doit correspondre à l'index du DataFrame."""
        result = compute_luminosity(df_train_minimal)
        pd.testing.assert_index_equal(result.index, df_train_minimal.index)

    def test_plusieurs_lignes(self):
        """Fonctionne correctement sur plusieurs lignes simultanément."""
        df     = pd.DataFrame({"Rs": [1.0, 2.0, 0.5], "Ts": [5772, 5772, 5772]})
        result = compute_luminosity(df)
        assert len(result) == 3
        # R double → L × 4 ; R halve → L / 4
        np.testing.assert_almost_equal(result.iloc[1] / result.iloc[0], 4.0, decimal=4)
        np.testing.assert_almost_equal(result.iloc[2] / result.iloc[0], 0.25, decimal=4)


# =============================================================================
# build_feature_matrix
# =============================================================================

class TestBuildFeatureMatrix:

    def test_shape_correcte(self, df_train_minimal):
        """La matrice X doit avoir (n_samples, 289) features."""
        X, feat_cols = build_feature_matrix(df_train_minimal)
        # 283 Rp + 6 astro (Ms, Mp, P, sma, i, L_stefan) = 289
        assert X.shape == (len(df_train_minimal), 289)
        assert len(feat_cols) == 289

    def test_dtype_float64(self, df_train_minimal):
        """La matrice X doit être en float64."""
        X, _ = build_feature_matrix(df_train_minimal)
        assert X.dtype == np.float64

    def test_dataframe_original_non_modifie(self, df_train_minimal):
        """build_feature_matrix ne doit pas modifier le DataFrame d'entrée."""
        cols_avant = set(df_train_minimal.columns)
        build_feature_matrix(df_train_minimal)
        cols_apres = set(df_train_minimal.columns)
        assert cols_avant == cols_apres, (
            f"Colonnes ajoutées au DataFrame original : {cols_apres - cols_avant}"
        )

    def test_eccentricite_exclue_par_defaut(self, df_train_minimal):
        """L'excentricité ne doit pas apparaître dans feat_cols par défaut."""
        _, feat_cols = build_feature_matrix(df_train_minimal, drop_eccentricity=True)
        assert "e" not in feat_cols

    def test_eccentricite_incluse_si_demande(self, df_train_minimal):
        """Avec drop_eccentricity=False, 'e' doit être dans feat_cols."""
        _, feat_cols = build_feature_matrix(df_train_minimal, drop_eccentricity=False)
        assert "e" in feat_cols
        # 289 + 1 (e) = 290 features
        assert len(feat_cols) == 290

    def test_l_stefan_presente_dans_feat_cols(self, df_train_minimal):
        """La feature dérivée L_stefan doit être dans feat_cols."""
        _, feat_cols = build_feature_matrix(df_train_minimal)
        assert "L_stefan" in feat_cols

    def test_rs_ts_absentes_de_feat_cols(self, df_train_minimal):
        """Rs et Ts doivent être remplacées par L_stefan et absentes de feat_cols."""
        _, feat_cols = build_feature_matrix(df_train_minimal)
        assert "Rs" not in feat_cols
        assert "Ts" not in feat_cols

    def test_rp_features_dans_les_283_premieres_colonnes(self, df_train_minimal):
        """Les 283 premières colonnes de X doivent correspondre aux Rp."""
        _, feat_cols = build_feature_matrix(df_train_minimal)
        rp_expected  = [f"Rp_{i}" for i in range(1, 284)]
        assert feat_cols[:283] == rp_expected

    def test_erreur_colonne_manquante(self, df_train_minimal):
        """
        Une ValueError explicite doit être levée si une colonne requise manque.
        Sans cette protection, on aurait un KeyError cryptique de pandas.
        """
        df_incomplet = df_train_minimal.drop(columns=["Ms"])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            build_feature_matrix(df_incomplet)

    def test_pas_de_nan_dans_X(self, df_train_minimal):
        """La matrice X ne doit contenir aucun NaN."""
        X, _ = build_feature_matrix(df_train_minimal)
        assert not np.isnan(X).any(), "NaN détectés dans la matrice X"


# =============================================================================
# build_target_matrix
# =============================================================================

class TestBuildTargetMatrix:

    def test_shape_correcte(self, df_train_minimal):
        """La matrice y doit avoir (n_samples, 283)."""
        y, wl_cols = build_target_matrix(df_train_minimal)
        assert y.shape == (len(df_train_minimal), 283)
        assert len(wl_cols) == 283

    def test_dtype_float64(self, df_train_minimal):
        """La matrice y doit être en float64."""
        y, _ = build_target_matrix(df_train_minimal)
        assert y.dtype == np.float64

    def test_noms_colonnes_wl(self, df_train_minimal):
        """Les colonnes doivent s'appeler wl_1, wl_2, ..., wl_283."""
        _, wl_cols = build_target_matrix(df_train_minimal)
        assert wl_cols[0]  == "wl_1"
        assert wl_cols[-1] == "wl_283"

    def test_erreur_colonnes_wl_manquantes(self, df_test_minimal):
        """
        df_test n'a pas les colonnes wl_i — une ValueError doit être levée.
        """
        with pytest.raises(ValueError, match="Colonnes cibles manquantes"):
            build_target_matrix(df_test_minimal)

    def test_pas_de_nan_dans_y(self, df_train_minimal):
        """La matrice y ne doit contenir aucun NaN."""
        y, _ = build_target_matrix(df_train_minimal)
        assert not np.isnan(y).any()


# =============================================================================
# compute_naive_stats
# =============================================================================

class TestComputeNaiveStats:

    def test_retourne_deux_floats(self, y_train_array):
        """La fonction doit retourner exactement 2 floats."""
        mean, sigma = compute_naive_stats(y_train_array)
        assert isinstance(mean,  float)
        assert isinstance(sigma, float)

    def test_valeurs_correctes(self, y_train_array):
        """mean et sigma doivent correspondre à np.mean et np.std."""
        mean, sigma = compute_naive_stats(y_train_array)
        np.testing.assert_almost_equal(mean,  float(np.mean(y_train_array)),  decimal=10)
        np.testing.assert_almost_equal(sigma, float(np.std(y_train_array)), decimal=10)

    def test_sigma_positif(self, y_train_array):
        """L'écart-type doit toujours être positif (ou nul pour un signal constant)."""
        _, sigma = compute_naive_stats(y_train_array)
        assert sigma >= 0

    def test_tableau_constant(self):
        """Pour un tableau constant, sigma doit être 0."""
        y    = np.full((10, 283), 0.1)
        _, sigma = compute_naive_stats(y)
        assert sigma == 0.0
