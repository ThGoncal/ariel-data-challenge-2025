"""
calibration.py — Corrections instrumentales Ariel Data Challenge 2025
======================================================================
Responsabilité : transformer le signal brut ADC en flux physique calibré.

Chaîne de traitement appliquée dans l'ordre :
    1. ADC_convert         : conversion Analogique → Numérique inverse
    2. CDS                 : Correlated Double Sampling (end - start)
    3. calibrate_and_clean : dark / flat / dead pixels / hot pixels (Numba)
    4. apply_linear_corr   : correction non-linéarité détecteur (Numba)
    5. bin_time            : binning temporel (Numba)

Les fonctions Numba (@njit) sont compilées à la première exécution
(JIT = Just In Time). Les appels suivants sont aussi rapides que du C.
numba.set_num_threads(1) est configuré ici pour éviter la sur-parallélisation
avec joblib (qui gère lui-même les workers au niveau des planètes).
"""

# === Scientific stack
import numpy as np

# === Numba
import numba
from numba import njit, prange

# On limite Numba à 1 thread car la parallélisation est gérée
# au niveau supérieur par joblib (une planète par worker).
# Si on laissait Numba utiliser tous les cœurs ET joblib aussi,
# on obtiendrait une sur-souscription (oversubscription) qui
# DÉGRADE les performances au lieu de les améliorer.
numba.set_num_threads(1)


# =============================================================================
# 1. CONVERSION ADC (Analog-to-Digital)
# =============================================================================

def ADC_convert(signal: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """
    Inverse la conversion Analogique → Numérique du détecteur.

    Le détecteur convertit une tension (flux de photons) en entier ADC
    Convention du script original (validée sur le challenge) :
    signal /= gain, puis signal += offset

    Cette correction est globale (un seul gain/offset pour tout le détecteur
    AIRS dans l'édition 2025 du challenge).

    Parameters
    ----------
    signal : ndarray, shape (n_time, n_pixels) ou (n_time, n_z, n_w)
        Signal brut en unités ADC (entiers)
    gain   : float, facteur de conversion ADC → tension
    offset : float, offset de conversion

    Returns
    -------
    signal : ndarray, même shape, en unités de flux (float64)
    """
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal


# =============================================================================
# 2. HOT PIXELS — fonctions Numba auxiliaires
# =============================================================================

@njit
def _replace_with_neighbors(arr: np.ndarray, index: int, num_neighbors: int = 1):
    """
    Remplace arr[index] par la moyenne de ses voisins temporels valides.

    Fonction interne appelée par correct_hot_pixels_1d.
    Le @njit sans parallel=True car cette fonction est appelée
    dans une boucle déjà parallélisée.

    Parameters
    ----------
    arr          : ndarray 1D — série temporelle d'un pixel
    index        : position du hot pixel à corriger
    num_neighbors: nombre de voisins de chaque côté à moyenner
    """
    n = len(arr)
    total = 0.0
    count = 0

    for i in range(1, num_neighbors + 1):
        if index - i >= 0:
            total += arr[index - i]
            count += 1
        if index + i < n:
            total += arr[index + i]
            count += 1

    if count > 0:
        arr[index] = total / count


@njit
def correct_hot_pixels_1d(
    data: np.ndarray,
    sigma_threshold: float = 5.0,
    num_neighbors: int = 1,
) -> np.ndarray:
    """
    Corrige les hot pixels temporels d'un pixel unique par sigma-clipping MAD.

    Algorithme :
        1. Calcule la médiane et la MAD (Median Absolute Deviation) de la série
        2. Tout point à plus de sigma_threshold × 1.4826 × MAD de la médiane
           est remplacé par la moyenne de ses voisins temporels

    Le facteur 1.4826 convertit la MAD en estimateur consistant de l'écart-type
    pour une distribution normale (σ ≈ 1.4826 × MAD).

    Cette fonction opère sur 1 pixel à la fois (série temporelle 1D).
    Elle est appelée en boucle parallèle dans calibrate_and_clean_cube.

    Parameters
    ----------
    data            : ndarray 1D (n_time,) — série temporelle d'un pixel
    sigma_threshold : seuil de détection (défaut 5σ)
    num_neighbors   : voisins temporels utilisés pour l'interpolation

    Returns
    -------
    corrected : ndarray 1D, même shape que data
    """
    n = len(data)
    corrected = data.copy()

    # --- Médiane (tri explicite car np.median non disponible dans njit)
    sorted_data = np.sort(data)
    median = sorted_data[n // 2]

    # --- MAD : médiane des écarts absolus à la médiane
    abs_dev    = np.abs(data - median)
    sorted_dev = np.sort(abs_dev)
    mad        = sorted_dev[n // 2]

    # Si MAD = 0, le signal est constant → aucun hot pixel possible
    if mad == 0:
        return corrected

    threshold = sigma_threshold * 1.4826 * mad

    for i in range(n):
        if np.abs(data[i] - median) > threshold:
            _replace_with_neighbors(corrected, i, num_neighbors=num_neighbors)

    return corrected


# =============================================================================
# 3. CALIBRATION COMPLÈTE DU CUBE (Numba parallèle)
# =============================================================================

@njit(parallel=True, fastmath=True)
def calibrate_and_clean_cube(
    signal: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    dead: np.ndarray,
    read: np.ndarray,
    dt: np.ndarray,
    sigma_threshold: float = 5.0,
    num_neighbors: int = 1,
) -> np.ndarray:
    """
    Calibre et nettoie un cube de données en une seule passe Numba.

    Enchaîne dans l'ordre pour chaque pixel (z, w) à chaque instant t :
        1. Soustraction du bruit de lecture (read noise)
        2. Soustraction du dark current (scalé par le temps d'intégration dt)
        3. Division par le flat field (réponse spatiale du détecteur)
        4. Mise à zéro des pixels morts (dead pixels mask)
    Puis, pour chaque pixel (z, w), correction des hot pixels temporels.

    La parallélisation est sur l'axe temporel (étape 1) et spatial (étape 2),
    ce qui maximise l'utilisation du cache CPU.

    Parameters
    ----------
    signal : ndarray (n_time, n_spatial, n_spectral)
        Cube CDS déjà normalisé par dt, en unités de flux
    dark   : ndarray (n_spatial, n_spectral) — courant d'obscurité par unité de temps
    flat   : ndarray (n_spatial, n_spectral) — champ plat normalisé
    dead   : ndarray bool (n_spatial, n_spectral) — True = pixel mort
    read   : ndarray (n_spatial, n_spectral) — bruit de lecture (offset fixe)
    dt     : ndarray (n_time,) — temps d'intégration par frame
    sigma_threshold : seuil sigma-clipping hot pixels
    num_neighbors   : voisins temporels pour interpolation hot pixels

    Returns
    -------
    calibrated : ndarray (n_time, n_spatial, n_spectral), float64
    """
    nt, nz, nw = signal.shape
    calibrated = np.empty_like(signal)

    # === Étape 1 : Calibration pixel par pixel, parallèle sur t
    for t in prange(nt):
        for z in range(nz):
            for w in range(nw):
                val = signal[t, z, w]

                # Bruit de lecture (offset fixe, indépendant du temps)
                val -= read[z, w]

                # Dark current (proportionnel au temps d'intégration)
                val -= dark[z, w] * dt[t]

                # Flat field + dead pixels
                if not dead[z, w]:
                    val /= flat[z, w]
                else:
                    val = 0.0  # pixel mort → on met à 0

                calibrated[t, z, w] = val

    # === Étape 2 : Correction hot pixels, parallèle sur z (spatial)
    # On traite chaque pixel (z, w) indépendamment sur sa série temporelle
    for z in prange(nz):
        for w in range(nw):
            calibrated[:, z, w] = correct_hot_pixels_1d(
                calibrated[:, z, w],
                sigma_threshold=sigma_threshold,
                num_neighbors=num_neighbors,
            )

    return calibrated


# =============================================================================
# 4. CORRECTION NON-LINÉARITÉ (Numba parallèle)
# =============================================================================

@njit(fastmath=True, parallel=True)
def apply_linear_corr(
    signal: np.ndarray,
    linear_corr: np.ndarray,
) -> np.ndarray:
    """
    Applique la correction de non-linéarité du détecteur par évaluation polynomiale.

    Le détecteur AIRS a une réponse non-linéaire au flux incident.
    La correction est modélisée par un polynôme d'ordre 5 dont les 6 coefficients
    sont stockés dans linear_corr[:, z, w] pour chaque pixel (z, w).

    Le polynôme est évalué par le schéma de Horner, qui minimise le nombre
    de multiplications (O(n) au lieu de O(n²)) et est numériquement stable.

    ⚠️ Les coefficients sont stockés en ordre croissant dans les parquets Ariel
    (linear_corr[0] = terme constant). Le `[::-1]` place le coefficient de plus
    haut degré en premier, comme l'exige le schéma de Horner ci-dessous
    (identique au script original).

    Parameters
    ----------
    signal      : ndarray (n_time, n_spatial, n_spectral)
    linear_corr : ndarray (6, n_spatial, n_spectral) — 6 coefficients polynomiaux

    Returns
    -------
    corrected : ndarray, même shape que signal
    """
    nt, nz, nw    = signal.shape
    n_coeff       = linear_corr.shape[0]
    corrected     = np.empty_like(signal)

    for z in prange(nz):
        for w in range(nw):
            # Inversion de l'ordre des coefficients (format Ariel → Horner)
            coeffs = linear_corr[::-1, z, w]

            for t in range(nt):
                x = signal[t, z, w]

                # Horner : y = (((c5)*x + c4)*x + ...)*x + c0
                y = coeffs[0]
                for k in range(1, n_coeff):
                    y = y * x + coeffs[k]

                corrected[t, z, w] = y

    return corrected


# =============================================================================
# 5. BINNING TEMPOREL (Numba parallèle)
# =============================================================================

@njit(parallel=True, fastmath=True)
def bin_time(signal: np.ndarray, binning: int) -> np.ndarray:
    """
    Réduit la résolution temporelle d'un cube 3D par moyennage.

    Le binning temporel sert à améliorer le rapport signal/bruit :
    en moyennant `binning` frames consécutives, on réduit le bruit
    de lecture d'un facteur √binning (pour du bruit gaussien).

    Le nombre de frames en sortie est nt // binning.
    Les frames restantes (nt % binning) sont ignorées — c'est
    le comportement standard pour ce challenge.

    Parameters
    ----------
    signal  : ndarray (n_time, n_spatial, n_spectral)
    binning : int, nombre de frames à moyenner

    Returns
    -------
    binned : ndarray (n_time // binning, n_spatial, n_spectral)
    """
    nt, nz, nw = signal.shape
    nt_bin     = nt // binning
    binned     = np.empty((nt_bin, nz, nw), dtype=signal.dtype)

    for t_bin in prange(nt_bin):
        for z in range(nz):
            for w in range(nw):
                s = 0.0
                for k in range(binning):
                    s += signal[t_bin * binning + k, z, w]
                binned[t_bin, z, w] = s / binning

    return binned


# =============================================================================
# 6. UTILITAIRE : Nettoyage des MaskedArray
# =============================================================================

def ensure_ndarray(signal: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Garantit que le signal est un np.ndarray standard compatible Numba.

    Les parquets Ariel peuvent parfois produire des numpy.ma.MaskedArray
    (tableaux avec masque) que Numba ne sait pas traiter. Cette fonction
    remplace les valeurs masquées par fill_value et retourne un ndarray
    standard.

    Parameters
    ----------
    signal     : ndarray ou MaskedArray
    fill_value : valeur de remplacement pour les entrées masquées

    Returns
    -------
    ndarray standard, jamais un MaskedArray
    """
    if isinstance(signal, np.ma.MaskedArray):
        return np.asarray(signal.filled(fill_value))
    return np.asarray(signal)
