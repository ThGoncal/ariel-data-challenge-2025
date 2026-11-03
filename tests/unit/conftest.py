"""
conftest.py — Fixtures partagées pytest
========================================
Ce fichier est chargé automatiquement par pytest avant tous les tests.
Les fixtures définies ici sont accessibles dans tous les fichiers de test
sans import explicite — pytest les injecte par nom dans les paramètres
des fonctions de test.

Stratégie des données de test :
    On génère de petits tableaux numpy synthétiques qui reproduisent
    EXACTEMENT la structure des données Ariel (dimensions, types, ranges)
    mais avec des tailles réduites pour que les tests s'exécutent en <1s.

    Dimensions réelles → dimensions de test :
        n_time     : ~36000 frames → 100 frames
        n_spatial  : 32 pixels     → 32 pixels (inchangé, c'est une constante)
        n_spectral : 283 λ         → 10 λ  (réduit pour la rapidité)
        n_samples  : 1100 planètes → 20 planètes
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# CONSTANTES DE TEST
# =============================================================================

N_TIME_TEST     = 100    # frames temporelles (réduit de 36000)
N_SPATIAL_TEST  = 32     # pixels spatiaux (identique au détecteur réel)
N_SPECTRAL_TEST = 10     # longueurs d'onde (réduit de 283)
N_SAMPLES_TEST  = 20     # planètes (réduit de 1100)
N_WL_FULL       = 283    # longueurs d'onde réelles pour les tests ML
SEED            = 42


# =============================================================================
# FIXTURES SIGNAL PROCESSING
# =============================================================================

@pytest.fixture
def rng():
    """Générateur de nombres aléatoires avec graine fixe — reproductibilité."""
    return np.random.default_rng(SEED)


@pytest.fixture
def signal_cube(rng):
    """
    Cube de signal 3D synthétique (n_time, n_spatial, n_spectral).

    Simule un signal de transit réaliste :
        - Flux de base ~1000 ADU (unités détecteur)
        - Bruit gaussien 1% (shot noise)
        - Creux de transit au centre (~5% de profondeur, 20% de la durée)

    Shape : (100, 32, 10)
    """
    nt, nz, nw = N_TIME_TEST, N_SPATIAL_TEST, N_SPECTRAL_TEST
    base_flux  = 1000.0

    # Signal de base avec bruit gaussien
    signal = rng.normal(loc=base_flux, scale=base_flux * 0.01, size=(nt, nz, nw))

    # Injection d'un transit : creux centré sur 20% du temps
    center  = nt // 2
    half_w  = nt // 10
    transit_depth = 0.05   # 5% de profondeur
    signal[center - half_w : center + half_w, :, :] *= (1 - transit_depth)

    return signal.astype(np.float64)


@pytest.fixture
def dark_frame(rng):
    """Carte de dark current (n_spatial, n_spectral). Valeurs ~1 ADU/s."""
    return rng.uniform(0.5, 1.5, size=(N_SPATIAL_TEST, N_SPECTRAL_TEST)).astype(np.float64)


@pytest.fixture
def flat_frame(rng):
    """Carte de flat field (n_spatial, n_spectral). Valeurs ~1.0 ± 5%."""
    return rng.uniform(0.95, 1.05, size=(N_SPATIAL_TEST, N_SPECTRAL_TEST)).astype(np.float64)


@pytest.fixture
def dead_mask():
    """Masque de pixels morts (n_spatial, n_spectral). ~2% de pixels morts."""
    mask = np.zeros((N_SPATIAL_TEST, N_SPECTRAL_TEST), dtype=bool)
    mask[0, 0] = True    # pixel mort coin supérieur gauche
    mask[5, 3] = True    # pixel mort aléatoire
    return mask


@pytest.fixture
def read_noise(rng):
    """Carte de bruit de lecture (n_spatial, n_spectral). Valeurs ~5 ADU."""
    return rng.uniform(4.0, 6.0, size=(N_SPATIAL_TEST, N_SPECTRAL_TEST)).astype(np.float64)


@pytest.fixture
def linear_corr_coeffs(rng):
    """
    Coefficients de correction non-linéaire (6, n_spatial, n_spectral).

    Le premier coefficient est ~1.0 (correction dominante linéaire),
    les autres sont ~0 (corrections d'ordre supérieur petites).
    """
    coeffs      = rng.uniform(-0.01, 0.01, size=(6, N_SPATIAL_TEST, N_SPECTRAL_TEST))
    coeffs[0]  += 1.0   # coefficient dominant ≈ 1
    return coeffs.astype(np.float64)


@pytest.fixture
def dt_array():
    """Tableau des temps d'intégration (n_time,). Valeurs ~0.1s."""
    return np.full(N_TIME_TEST, 0.1, dtype=np.float64)


# =============================================================================
# FIXTURES ML
# =============================================================================

@pytest.fixture
def df_train_minimal(rng):
    """
    DataFrame minimal simulant train_with_spectra.csv.

    Contient toutes les colonnes attendues par build_feature_matrix()
    et build_target_matrix() avec N_SAMPLES_TEST=20 lignes.
    """
    n = N_SAMPLES_TEST

    data = {
        "planet_id": np.arange(1, n + 1),
        # Paramètres stellaires / orbitaux
        "Rs": rng.uniform(0.5, 2.0, n),     # rayon stellaire (R_sun)
        "Ms": rng.uniform(0.5, 2.0, n),     # masse stellaire (M_sun)
        "Ts": rng.uniform(3000, 7000, n),   # température effective (K)
        "Mp": rng.uniform(0.1, 10.0, n),    # masse planétaire (M_jup)
        "e":  np.zeros(n),                  # excentricité = 0 pour Ariel 2025
        "P":  rng.uniform(1.0, 30.0, n),    # période orbitale (jours)
        "sma": rng.uniform(0.01, 1.0, n),   # demi-grand axe (AU)
        "i":  rng.uniform(80, 90, n),       # inclinaison (degrés)
    }

    # Features Rp_1..283 (spectres mesurés)
    for i in range(1, N_WL_FULL + 1):
        data[f"Rp_{i}"] = rng.uniform(0.05, 0.15, n)

    # Targets wl_1..283 (spectres de transmission vrais)
    for i in range(1, N_WL_FULL + 1):
        data[f"wl_{i}"] = rng.uniform(0.08, 0.12, n)

    return pd.DataFrame(data)


@pytest.fixture
def df_test_minimal(rng):
    """DataFrame minimal simulant test_with_spectra.csv (sans colonnes wl_i)."""
    n = N_SAMPLES_TEST // 2

    data = {
        "planet_id": np.arange(2000, 2000 + n),
        "Rs":  rng.uniform(0.5, 2.0, n),
        "Ms":  rng.uniform(0.5, 2.0, n),
        "Ts":  rng.uniform(3000, 7000, n),
        "Mp":  rng.uniform(0.1, 10.0, n),
        "e":   np.zeros(n),
        "P":   rng.uniform(1.0, 30.0, n),
        "sma": rng.uniform(0.01, 1.0, n),
        "i":   rng.uniform(80, 90, n),
    }
    for i in range(1, N_WL_FULL + 1):
        data[f"Rp_{i}"] = rng.uniform(0.05, 0.15, n)

    return pd.DataFrame(data)


@pytest.fixture
def y_train_array(rng):
    """Matrice de targets (N_SAMPLES_TEST, 283) en valeurs physiques réalistes."""
    return rng.uniform(0.08, 0.12, size=(N_SAMPLES_TEST, N_WL_FULL)).astype(np.float64)


@pytest.fixture
def y_pred_mean(rng, y_train_array):
    """Prédictions moyennes légèrement bruitées autour des vraies valeurs."""
    noise = rng.normal(0, 0.005, size=y_train_array.shape)
    return np.clip(y_train_array + noise, a_min=1e-6, a_max=None)


@pytest.fixture
def y_pred_std(rng, y_train_array):
    """Écarts-types prédits réalistes (~5% de la valeur)."""
    return rng.uniform(0.001, 0.01, size=y_train_array.shape)


@pytest.fixture
def planet_ids(df_train_minimal):
    """Array des planet_id du set d'entraînement minimal."""
    return df_train_minimal["planet_id"].values
