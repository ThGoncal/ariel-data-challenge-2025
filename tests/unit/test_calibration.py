"""
test_calibration.py — Tests unitaires pour src/signal_processing/calibration.py
=================================================================================
Stratégie de test :
    Chaque fonction est testée avec des valeurs synthétiques dont
    le résultat EXACT est connu analytiquement.
    On ne teste pas les performances (vitesse Numba) mais la CORRECTION
    mathématique des transformations.

Règles d'écriture des tests :
    - Un test = une assertion précise
    - Nom explicite : test_<fonction>_<scénario>_<résultat_attendu>
    - Les tests sont indépendants (pas d'état partagé entre tests)
    - Pas de fichiers I/O — uniquement des arrays en mémoire
"""

import numpy as np
import pytest

from src.signal_processing.calibration import (
    ADC_convert,
    correct_hot_pixels_1d,
    calibrate_and_clean_cube,
    apply_linear_corr,
    bin_time,
    ensure_ndarray,
)


# =============================================================================
# ADC_convert
# =============================================================================

class TestADCConvert:
    """
    Convention du script original : signal / gain + offset.
    """

    def test_identite_gain1_offset0(self):
        """Avec gain=1 et offset=0, le signal doit être inchangé."""
        signal = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = ADC_convert(signal.copy(), gain=1.0, offset=0.0)
        np.testing.assert_array_almost_equal(result, signal)

    def test_soustraction_offset(self):
        """L'offset est ajouté après la division par le gain."""
        signal = np.array([[10.0, 20.0]])
        result = ADC_convert(signal.copy(), gain=1.0, offset=5.0)
        expected = np.array([[15.0, 25.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiplication_gain(self):
        """Le signal est divisé par le gain."""
        signal = np.array([[2.0, 4.0]])
        result = ADC_convert(signal.copy(), gain=3.0, offset=0.0)
        expected = np.array([[2.0/3.0, 4.0/3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_gain_et_offset_combines(self):
        """Test avec gain et offset non triviaux simultanément."""
        signal   = np.array([[110.0]])
        gain     = 2.0
        offset   = 10.0
        result   = ADC_convert(signal.copy(), gain=gain, offset=offset)
        expected = signal / gain + offset    # 55 + 10 = 65
        np.testing.assert_array_almost_equal(result, expected)

    def test_conversion_en_float64(self):
        """Le résultat doit toujours être float64, même si l'entrée est int."""
        signal = np.array([[100, 200]], dtype=np.int32)
        result = ADC_convert(signal, gain=1.0, offset=0.0)
        assert result.dtype == np.float64

    def test_shape_preservee(self):
        """La shape du tableau d'entrée doit être préservée."""
        signal = np.ones((50, 32, 10))
        result = ADC_convert(signal, gain=2.0, offset=1.0)
        assert result.shape == signal.shape

    def test_pas_de_mutation_inplace(self):
        """ADC_convert ne doit pas modifier le tableau original."""
        signal   = np.array([[100.0, 200.0]])
        original = signal.copy()
        ADC_convert(signal, gain=2.0, offset=5.0)
        # signal original inchangé car on fait signal.astype() qui crée une copie
        np.testing.assert_array_equal(signal, original)


# =============================================================================
# correct_hot_pixels_1d
# =============================================================================

class TestCorrectHotPixels:
    """
    La correction détecte les outliers par MAD sigma-clipping
    et les remplace par la moyenne de leurs voisins temporels.
    """

    def test_serie_constante_inchangee(self):
        """Une série constante ne contient aucun hot pixel — pas de modification."""
        data   = np.ones(100, dtype=np.float64)
        result = correct_hot_pixels_1d(data.copy(), sigma_threshold=5.0)
        np.testing.assert_array_almost_equal(result, data)

    def test_hot_pixel_central_corrige(self):
        """Un hot pixel évident au centre doit être remplacé par ses voisins."""
        data        = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(50)  # MAD != 0
        data[25]    = 1000.0    # hot pixel flagrant (σ > 5)
        result      = correct_hot_pixels_1d(data.copy(), sigma_threshold=5.0)

        # Le hot pixel doit être proche de 1.0 (ses voisins)
        assert abs(result[25] - 1.0) < 0.1, (
            f"Hot pixel non corrigé : result[25] = {result[25]:.4f}, attendu ≈ 1.0"
        )

    def test_points_normaux_non_touches(self):
        """Les points dans la distribution normale ne doivent pas être modifiés."""
        rng     = np.random.default_rng(42)
        data    = rng.normal(loc=100.0, scale=1.0, size=200)
        data    = data.astype(np.float64)
        # Injection d'un seul hot pixel
        data[100] = 1e6
        result  = correct_hot_pixels_1d(data.copy(), sigma_threshold=5.0)

        # Tous les points sauf data[100] doivent être inchangés
        mask             = np.ones(len(data), dtype=bool)
        mask[100]        = False
        np.testing.assert_array_almost_equal(result[mask], data[mask])

    def test_mad_zero_retourne_serie_inchangee(self):
        """Si MAD=0 (signal constant), la fonction retourne la série inchangée."""
        data   = np.full(50, 5.0, dtype=np.float64)
        result = correct_hot_pixels_1d(data.copy())
        np.testing.assert_array_equal(result, data)

    def test_hot_pixel_en_bord_gauche(self):
        """Un hot pixel au premier indice n'a qu'un seul voisin disponible."""
        data    = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(30)  # MAD != 0
        data[0] = 9999.0
        result  = correct_hot_pixels_1d(data.copy(), sigma_threshold=5.0)
        # Voisin de data[0] = data[1] = 1.0
        assert abs(result[0] - 1.0) < 0.1

    def test_retourne_meme_shape(self):
        """La shape de la série doit être préservée."""
        data   = np.random.rand(100).astype(np.float64)
        result = correct_hot_pixels_1d(data.copy())
        assert result.shape == data.shape


# =============================================================================
# calibrate_and_clean_cube
# =============================================================================

class TestCalibrateAndCleanCube:
    """
    Tests de la calibration complète du cube :
    soustraction read + dark, division flat, mise à zéro pixels morts,
    correction hot pixels.
    """

    def test_soustraction_read_noise(
        self, signal_cube, dark_frame, flat_frame, dead_mask, read_noise, dt_array
    ):
        """
        Vérifie que le bruit de lecture est soustrait.
        Avec dark=0, flat=1, dead=False, le résultat doit être signal - read.
        """
        nt, nz, nw = signal_cube.shape
        dark_zero  = np.zeros((nz, nw))
        flat_one   = np.ones((nz, nw))
        dead_false = np.zeros((nz, nw), dtype=bool)

        result = calibrate_and_clean_cube(
            signal_cube.copy(), dark_zero, flat_one, dead_false,
            read_noise, dt_array
        )

        # Pour le pixel (0,0) au temps 0, sans hot pixel :
        # expected = signal[0,0,0] - read[0,0] - dark[0,0]*dt[0] / flat[0,0]
        # avec dark=0, flat=1 → expected = signal[0,0,0] - read[0,0]
        expected_00 = signal_cube[0, 0, 0] - read_noise[0, 0]
        # Note : la correction hot pixels peut légèrement modifier la valeur
        # On vérifie juste que le résultat est dans l'ordre de grandeur attendu
        assert abs(result[0, 0, 0] - expected_00) < signal_cube[0, 0, 0] * 0.1

    def test_pixels_morts_mis_a_zero(
        self, signal_cube, dark_frame, flat_frame, dead_mask, read_noise, dt_array
    ):
        """Les pixels marqués dead=True doivent être mis à 0 dans le résultat."""
        result = calibrate_and_clean_cube(
            signal_cube.copy(), dark_frame, flat_frame,
            dead_mask, read_noise, dt_array
        )
        # dead_mask[0,0] = True → result[:,0,0] doit être 0 partout
        dead_z, dead_w = np.where(dead_mask)
        for z, w in zip(dead_z, dead_w):
            np.testing.assert_array_equal(
                result[:, z, w], 0.0,
                err_msg=f"Pixel mort ({z},{w}) non mis à 0"
            )

    def test_shape_preservee(
        self, signal_cube, dark_frame, flat_frame, dead_mask, read_noise, dt_array
    ):
        """La shape du cube doit être préservée après calibration."""
        result = calibrate_and_clean_cube(
            signal_cube.copy(), dark_frame, flat_frame,
            dead_mask, read_noise, dt_array
        )
        assert result.shape == signal_cube.shape

    def test_output_dtype_float64(
        self, signal_cube, dark_frame, flat_frame, dead_mask, read_noise, dt_array
    ):
        """Le résultat doit être en float64."""
        result = calibrate_and_clean_cube(
            signal_cube.copy(), dark_frame, flat_frame,
            dead_mask, read_noise, dt_array
        )
        assert result.dtype == np.float64


# =============================================================================
# apply_linear_corr
# =============================================================================

class TestApplyLinearCorr:
    """
    La correction non-linéaire applique un polynôme par pixel via Horner.
    Avec des coefficients [1, 0, 0, 0, 0, 0], le résultat = signal (identité).
    """

    def test_coefficients_identite(self, signal_cube):
        """
        Avec linear_corr = [[1], [0], [0], [0], [0], [0]], le polynôme
        est p(x) = x → signal inchangé.
        """
        nt, nz, nw = signal_cube.shape
        # Coefficients du polynôme identité : [1, 0, 0, 0, 0, 0]
        # Rappel : les coefficients sont en ordre décroissant dans Ariel
        # → on stocke [0, 0, 0, 0, 0, 1] pour que [::-1] donne [1, 0, ...]
        # et que Horner évalue c0 + c1*x + ... = x
        linear_corr = np.zeros((6, nz, nw))
        linear_corr[1, :, :] = 1.0   # ordre croissant : [c0, c1, ...] -> p(x) = x

        result = apply_linear_corr(signal_cube.copy(), linear_corr)
        np.testing.assert_array_almost_equal(result, signal_cube, decimal=5)

    def test_shape_preservee(self, signal_cube, linear_corr_coeffs):
        """La shape doit être préservée."""
        result = apply_linear_corr(signal_cube.copy(), linear_corr_coeffs)
        assert result.shape == signal_cube.shape

    def test_coefficients_constants(self, signal_cube):
        """
        Avec linear_corr = [[0], ..., [0], [c]], le polynôme est p(x) = c
        (constante). Résultat = c pour tous les pixels.
        """
        nt, nz, nw = signal_cube.shape
        c           = 5.0
        linear_corr = np.zeros((6, nz, nw))
        # Après [::-1], le premier coeff (ordre 0 de Horner) vaut c
        linear_corr[0, :, :] = c   # p(x) = c (constante, terme d'ordre 0)

        # Note : avec Horner, y = c0 + x*(c1 + x*(...))
        # Si tous les coeffs sauf c0=c sont 0, y = c
        # Mais notre coeffs est [0,0,0,0,0,c] → [::-1] = [c,0,0,0,0,0]
        # Horner : y = c + x*(0 + x*(0 + ...)) = c
        result = apply_linear_corr(signal_cube.copy(), linear_corr)
        np.testing.assert_array_almost_equal(result, np.full_like(result, c), decimal=4)


# =============================================================================
# bin_time
# =============================================================================

class TestBinTime:
    """
    Le binning temporel moyenne N frames consécutives.
    On vérifie la shape, les valeurs et les cas limites.
    """

    def test_binning_2_shape(self, signal_cube):
        """Avec binning=2, le temps est divisé par 2."""
        nt, nz, nw = signal_cube.shape
        result     = bin_time(signal_cube.astype(np.float64), binning=2)
        assert result.shape == (nt // 2, nz, nw)

    def test_binning_5_shape(self, signal_cube):
        """Avec binning=5, le temps est divisé par 5."""
        nt, nz, nw = signal_cube.shape
        result     = bin_time(signal_cube.astype(np.float64), binning=5)
        assert result.shape == (nt // 5, nz, nw)

    def test_valeurs_moyennees_correctement(self):
        """Vérifie que les valeurs sont bien la moyenne des frames groupées."""
        # Signal simple : frames 0,1 → [1,3] ; moyenne = 2
        nt, nz, nw = 4, 2, 3
        signal     = np.zeros((nt, nz, nw), dtype=np.float64)
        signal[0]  = 1.0
        signal[1]  = 3.0
        signal[2]  = 5.0
        signal[3]  = 7.0

        result = bin_time(signal, binning=2)
        # frame 0 binné = moyenne(1, 3) = 2
        # frame 1 binné = moyenne(5, 7) = 6
        np.testing.assert_array_almost_equal(result[0], np.full((nz, nw), 2.0))
        np.testing.assert_array_almost_equal(result[1], np.full((nz, nw), 6.0))

    def test_binning_1_identite(self, signal_cube):
        """Avec binning=1, le résultat doit être identique à l'entrée."""
        signal = signal_cube.astype(np.float64)
        result = bin_time(signal, binning=1)
        np.testing.assert_array_almost_equal(result, signal)

    def test_frames_restantes_ignorees(self):
        """Les frames restantes (nt % binning != 0) sont ignorées silencieusement."""
        # nt=10, binning=3 → 10//3 = 3 bins, 1 frame ignorée
        signal = np.ones((10, 2, 2), dtype=np.float64)
        result = bin_time(signal, binning=3)
        assert result.shape[0] == 3


# =============================================================================
# ensure_ndarray
# =============================================================================

class TestEnsureNdarray:
    """ensure_ndarray convertit les MaskedArray en ndarray standard."""

    def test_ndarray_standard_inchange(self):
        """Un ndarray standard passe sans modification."""
        arr    = np.array([1.0, 2.0, 3.0])
        result = ensure_ndarray(arr)
        np.testing.assert_array_equal(result, arr)
        assert not isinstance(result, np.ma.MaskedArray)

    def test_masked_array_converti(self):
        """Un MaskedArray est converti en ndarray avec fill_value."""
        data   = np.array([1.0, 2.0, 3.0])
        mask   = np.array([False, True, False])   # data[1] est masqué
        ma     = np.ma.MaskedArray(data, mask=mask)

        result = ensure_ndarray(ma, fill_value=0.0)

        assert not isinstance(result, np.ma.MaskedArray)
        assert result[0] == 1.0
        assert result[1] == 0.0    # valeur masquée remplacée par fill_value
        assert result[2] == 3.0

    def test_fill_value_personnalise(self):
        """Le fill_value personnalisé est utilisé pour les valeurs masquées."""
        data   = np.array([10.0, 20.0])
        mask   = np.array([True, False])
        ma     = np.ma.MaskedArray(data, mask=mask)
        result = ensure_ndarray(ma, fill_value=-999.0)
        assert result[0] == -999.0
