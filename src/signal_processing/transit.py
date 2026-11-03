"""
transit.py — Extraction du spectre de transit Ariel Data Challenge 2025
========================================================================
Responsabilité : calculer le rayon relatif Rp/Rs à partir d'un cube
de signal calibré, pour chaque longueur d'onde.

Deux méthodes disponibles :
    - "median" : rapide, robuste, recommandée pour la production
                 (implémentée en Numba, parallèle sur les longueurs d'onde)
    - "fit"    : plus précise sur des signaux propres, mais 10× plus lente
                 (fit d'une super-gaussienne inversée par scipy.optimize)

Physique du transit :
    Lors du transit d'une exoplanète devant son étoile, le flux stellaire
    reçu diminue proportionnellement à (Rp/Rs)² — le rapport des surfaces.
    En mesurant la profondeur du transit ΔF = 1 - F_in/F_out, on retrouve :
        Rp/Rs = √(ΔF)
    On répète ce calcul pour chaque longueur d'onde → spectre de transmission.
"""

# === Standard library
import logging
from warnings import catch_warnings

# === Scientific stack
import numpy as np
from scipy.optimize import curve_fit

# === Numba
from numba import njit, prange

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CALCUL DU FLUX INTÉGRÉ (Numba)
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_flux(signal: np.ndarray) -> np.ndarray:
    """
    Intègre le signal sur l'axe spatial pour obtenir la courbe de lumière
    par longueur d'onde.

    Pour chaque longueur d'onde wl et chaque instant t, on somme les
    contributions de tous les pixels spatiaux z. On obtient ainsi le flux
    total reçu par le détecteur à chaque pas de temps.

    Parameters
    ----------
    signal : ndarray (n_time, n_spatial, n_spectral)

    Returns
    -------
    flux : ndarray (n_spectral, n_time)
        Courbe de lumière pour chaque longueur d'onde.
        L'axe spectral est mis en premier pour faciliter l'itération
        par longueur d'onde dans calcul_Rp_fit.
    """
    n_time, n_spat, n_wl = signal.shape
    flux = np.zeros((n_wl, n_time))

    # Parallèle sur les longueurs d'onde (axe indépendant)
    for wl in prange(n_wl):
        for t in range(n_time):
            total = 0.0
            for z in range(n_spat):
                total += signal[t, z, wl]
            flux[wl, t] = total

    return flux


# =============================================================================
# 2. MÉTHODE "MEDIAN" — Rp/Rs par médiane in/out transit (Numba)
# =============================================================================

@njit(parallel=True, fastmath=True)
def calcul_Rp_median(signal: np.ndarray) -> np.ndarray:
    """
    Calcule Rp/Rs par comparaison des médianes in-transit et out-of-transit.

    Méthode robuste et rapide, parallélisée sur les longueurs d'onde.

    Hypothèse simplificatrice : le transit est centré dans la série temporelle
    (ce qui est garanti par la simulation Ariel).

    Algorithme :
        - F_in  : médiane du flux dans une fenêtre de 10% autour du centre
        - F_out : médiane du flux sur le reste (avant + après transit)
        - ΔF    = max(0, 1 - F_in / F_out)   [clippé à 0 pour éviter les négatifs]
        - Rp/Rs = √(ΔF)

    Parameters
    ----------
    signal : ndarray (n_time, n_spatial, n_spectral), calibré et binné

    Returns
    -------
    Rp : ndarray (n_spectral,)
        Spectre Rp/Rs, une valeur par longueur d'onde
    """
    n_time, n_spat, n_wl = signal.shape
    Rp = np.zeros(n_wl, dtype=np.float64)

    # Somme spatiale → courbes de lumière (n_time, n_wl)
    flux = signal.sum(axis=1)

    # Fenêtre "in transit" : 10% des frames autour du centre
    mid    = n_time // 2
    window = max(n_time // 10, 1)  # au moins 1 frame

    for wl in prange(n_wl):
        f = flux[:, wl]

        f_in  = f[mid - window : mid + window]
        f_out = np.concatenate((f[:mid - window], f[mid + window:]))

        F_in  = np.median(f_in)
        F_out = np.median(f_out)

        if F_out > 0 and F_in > 0:
            depth    = max(1.0 - (F_in / F_out), 0.0)
            Rp[wl]   = np.sqrt(depth)
        else:
            Rp[wl] = 0.0

    return Rp


# =============================================================================
# 3. MÉTHODE "FIT" — Super-gaussienne inversée (scipy)
# =============================================================================

def _super_gaussienne_inversee(
    x: np.ndarray,
    bas: float,
    haut: float,
    centre: float,
    largeur: float,
    n: float,
) -> np.ndarray:
    """
    Modèle analytique d'une courbe de transit : super-gaussienne inversée.

    La super-gaussienne inversée est une forme générale qui peut modéliser
    des transits plus "carrés" (n élevé) ou plus arrondis (n=2 → gaussienne).
    Elle est définie comme :
        y(x) = haut - (haut - bas) × exp(-|x - centre|ⁿ / (2 × largeurⁿ))

    Parameters
    ----------
    x       : ndarray, axe temporel (indices des frames)
    bas     : flux minimal (au fond du transit)
    haut    : flux hors transit (niveau de référence stellaire)
    centre  : position temporelle du centre du transit
    largeur : demi-largeur caractéristique du transit
    n       : ordre de la super-gaussienne (n=2 → gaussienne standard)

    Returns
    -------
    y : ndarray, même shape que x
    """
    return haut - (haut - bas) * np.exp(
        -(np.abs(x - centre) / largeur) ** n / 2
    )


def _fit_super_gaussienne(
    x: np.ndarray, y: np.ndarray
) -> tuple | None:
    """
    Ajuste une super-gaussienne inversée sur la courbe de lumière.

    Retourne (params, cov) si le fit converge, None sinon.
    Les bornes physiques garantissent des valeurs cohérentes :
    flux positif, centre dans la série, largeur > 1 frame, ordre ∈ [0.5, 20].

    Parameters
    ----------
    x : ndarray 1D — axe temporel (indices)
    y : ndarray 1D — flux observé

    Returns
    -------
    (params, cov) ou None si échec de convergence
    """
    p0 = [
        np.min(y),      # bas    : estimation initiale = minimum observé
        np.max(y),      # haut   : estimation initiale = maximum observé
        len(x) / 2,     # centre : milieu de la série
        len(x) / 10,    # largeur: ~10% de la durée totale
        2,              # n      : gaussienne standard comme point de départ
    ]
    bounds = (
        [0,          0,          0,      1,       0.5],  # bornes inférieures
        [np.inf, np.inf, len(x),  len(x), 20  ],        # bornes supérieures
    )

    with catch_warnings():
        try:
            params, cov = curve_fit(
                _super_gaussienne_inversee,
                x, y,
                p0=p0,
                bounds=bounds,
                maxfev=10_000,
            )
            return params, cov
        except Exception as exc:
            logger.debug(f"Fit super-gaussienne échoué : {exc}")
            return None


def calcul_Rp_fit(
    signal: np.ndarray,
    planet_id: str = "?",
) -> np.ndarray:
    """
    Calcule Rp/Rs par fit de super-gaussienne inversée (méthode scipy).

    Plus précise que la méthode médiane sur des signaux propres,
    mais ~10× plus lente (pas de parallélisation Numba).
    Utiliser pour des analyses ponctuelles ou la validation.

    Parameters
    ----------
    signal    : ndarray (n_time, n_spatial, n_spectral), calibré et binné
    planet_id : str, pour les messages de log

    Returns
    -------
    Rp : ndarray (n_spectral,)
    """
    from calibration import ensure_ndarray
    signal    = ensure_ndarray(signal)
    flux_arr  = compute_flux(signal)       # (n_wl, n_time)

    n_wl   = signal.shape[2]
    t      = np.arange(signal.shape[0])
    Rp     = []

    for wl in range(n_wl):
        lc     = flux_arr[wl, :]
        result = _fit_super_gaussienne(t, lc)

        if result is not None:
            params, _ = result
            bas, haut, *_ = params
            ratio = bas / haut
            Rp.append(np.sqrt(max(1.0 - ratio, 0.0)))
        else:
            Rp.append(0.0)

    return np.array(Rp)


# =============================================================================
# 4. EXTRACTION Rp FGS1 (photométrique)
# =============================================================================

def calcul_Rp_fgs1(signal: np.ndarray) -> np.ndarray:
    """
    Calcule Rp/Rs pour l'instrument FGS1 (photométrique intégré).

    FGS1 n'est pas un spectromètre : il intègre tout le flux reçu
    sur son détecteur 32×32. On obtient donc une seule valeur Rp
    (pas un spectre), qui correspond à la mesure photométrique
    de la profondeur du transit intégré en longueur d'onde.

    On utilise un fit de super-gaussienne sur la courbe de lumière
    totale (somme de tous les pixels du détecteur).

    Parameters
    ----------
    signal : ndarray (n_time, n_spatial_y, n_spatial_x)
        Cube FGS1 calibré et binné.
        Note : les deux dernières dimensions sont spatiales (pas spectrales).

    Returns
    -------
    Rp : ndarray (1,)
        Une seule valeur Rp/Rs pour l'ensemble du canal photométrique.
        Retourne [nan] si le fit échoue.
    """
    # Somme de tous les pixels du détecteur → courbe de lumière scalaire
    flux = signal.sum(axis=(1, 2))   # (n_time,)
    t    = np.arange(flux.shape[0])

    result = _fit_super_gaussienne(t, flux)

    if result is None:
        logger.warning("FGS1 : fit super-gaussienne échoué → Rp = NaN")
        return np.array([np.nan])

    params, _ = result
    bas, haut, *_ = params
    depth = max(1.0 - (bas / haut), 0.0)
    return np.array([np.sqrt(depth)])
