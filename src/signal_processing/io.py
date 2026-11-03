"""
io.py — Chargement des données Ariel Data Challenge 2025
=========================================================
Responsabilité unique : lire les fichiers parquet du dataset
et les retourner sous forme de ndarrays prêts pour le traitement.

Aucune transformation du signal ici — uniquement de la lecture.
"""

# === Standard library
import os
import logging
from pathlib import Path

# === Scientific stack
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES INSTRUMENTALES
# =============================================================================

# Dimensions des détecteurs
AIRS_SHAPE_SPATIAL   = 32   # pixels spatiaux AIRS (axe y)
AIRS_SHAPE_SPECTRAL  = 356  # pixels spectraux AIRS (axe longueur d'onde, avant coupe)
FGS1_SHAPE           = 32   # pixels FGS1 (détecteur carré 32×32)

# Nombre de coefficients de correction non-linéaire
N_LINEAR_CORR_COEFFS = 6


# =============================================================================
# INDEXATION DES LONGUEURS D'ONDE
# =============================================================================

def wavelength_index(wavelengths_path: str | Path, axis_info_path: str | Path) -> np.ndarray:
    """
    Calcule les indices des longueurs d'onde utiles (283 valeurs = FGS1 + 282 AIRS)
    dans le peigne complet AIRS (356 valeurs).

    Le fichier wavelengths.csv contient les 283 longueurs d'onde
    scientifiquement exploitables. Le fichier axis_info.parquet contient
    le peigne complet des 356 pixels spectraux du détecteur AIRS.
    Cette fonction retourne les positions (indices) des 283 valeurs
    utiles dans le tableau de 356 — ces indices servent ensuite
    à découper les cubes de signal (cut_inf, cut_sup).

    Parameters
    ----------
    wavelengths_path : str | Path
        Chemin vers wavelengths.csv  (283 longueurs d'onde utiles)
    axis_info_path : str | Path
        Chemin vers axis_info.parquet (356 longueurs d'onde complètes + temps)

    Returns
    -------
    wl_idx : ndarray (shape ≈ 283,)
        Indices dans le peigne complet trié correspondant aux
        longueurs d'onde utiles.
    """
    # --- Chargement des 283 longueurs d'onde utiles
    df_wavelengths = pd.read_csv(wavelengths_path)
    wl_useful = df_wavelengths.values.flatten()

    # --- Chargement du peigne complet AIRS (356 valeurs)
    df_axis_info = pd.read_parquet(axis_info_path)
    wl_full = df_axis_info["AIRS-CH0-axis2-um"].dropna().values

    # --- Arrondi pour éviter les erreurs de comparaison flottante
    decimals = 6
    wl_useful_round = np.round(wl_useful, decimals)
    wl_full_round   = np.round(wl_full,   decimals)

    # --- Tri croissant du peigne complet
    sort_idx   = np.argsort(wl_full_round)
    wl_full_sorted = wl_full_round[sort_idx]

    # --- Recherche vectorisée : positions des 283 valeurs dans les 356
    wl_idx = np.where(np.isin(wl_full_sorted, wl_useful_round))[0]

    logger.info(
        f"wavelength_index : {len(wl_idx)} longueurs d'onde utiles "
        f"trouvées sur {len(wl_full)} disponibles "
        f"(indices {wl_idx[0]}..{wl_idx[-1]})"
    )
    return wl_idx


# =============================================================================
# CHARGEMENT DU SIGNAL BRUT
# =============================================================================

def load_signal(
    planet_id: str,
    instrument: str,
    base_path: str | Path,
    base_folder: str,
    signal_filename: str,
) -> np.ndarray:
    """
    Charge le signal brut d'une planète pour un instrument donné.

    Le fichier parquet est lu colonne par colonne — chaque colonne
    correspond à un pixel du détecteur aplati. On retourne un ndarray
    2D (n_time, n_pixels) en float64, avant tout reshape.
    Le reshape (n_time, n_spatial, n_spectral) est fait en aval
    dans pipeline.py pour ne pas mélanger les responsabilités.

    Parameters
    ----------
    planet_id       : identifiant de la planète (ex: "100")
    instrument      : "AIRS" ou "FGS1"
    base_path       : racine des données (ex: Path("data/raw"))
    base_folder     : "train" ou "test"
    signal_filename : nom du fichier parquet (ex: "AIRS-CH0_signal_0.parquet")

    Returns
    -------
    signal : ndarray (n_time, n_pixels), dtype float64
    """
    signal_path = Path(base_path) / base_folder / planet_id / signal_filename

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Signal introuvable : {signal_path}\n"
            f"Vérifiez que la planète {planet_id} a bien été téléchargée."
        )

    df = pd.read_parquet(signal_path)
    signal = df.values.astype(np.float64)

    logger.debug(
        f"load_signal | planète={planet_id} instrument={instrument} "
        f"shape={signal.shape} fichier={signal_filename}"
    )
    return signal


# =============================================================================
# CHARGEMENT DES FICHIERS DE CALIBRATION
# =============================================================================

def load_calib(
    planet_id: str,
    base_path: str | Path,
    base_folder: str,
    calibration_dir: str,
    instrument: str = "AIRS",
    cut_inf: int = None,
    cut_sup: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et reshape les 5 fichiers de calibration d'une planète.

    Les fichiers de calibration sont stockés aplatis dans les parquets.
    Cette fonction les charge, les convertit en float64/bool selon le type,
    puis les reshape aux dimensions du détecteur correspondant.

    Pour AIRS, la coupe spectrale [cut_inf:cut_sup] est appliquée
    immédiatement — on ne garde que les 283 longueurs d'onde utiles.
    Pour FGS1, le détecteur est carré (32×32), pas de coupe spectrale.

    Parameters
    ----------
    planet_id       : identifiant de la planète
    base_path       : racine des données
    base_folder     : "train" ou "test"
    calibration_dir : ex "AIRS-CH0_calibration_0" ou "FGS1_calibration_0"
    instrument      : "AIRS" ou "FGS1"
    cut_inf, cut_sup: bornes de la coupe spectrale (AIRS uniquement)
                      obtenues via wavelength_index()

    Returns
    -------
    dark        : ndarray (n_spatial, n_spectral)  courant d'obscurité
    flat        : ndarray (n_spatial, n_spectral)  champ plat
    dead        : ndarray (n_spatial, n_spectral)  bool, pixels morts
    read        : ndarray (n_spatial, n_spectral)  bruit de lecture
    linear_corr : ndarray (6, n_spatial, n_spectral) coefficients non-linéarité
    """
    calib_dir = Path(base_path) / base_folder / planet_id / calibration_dir

    # --- Chemins des 5 fichiers de calibration
    paths = {
        "dark":        calib_dir / "dark.parquet",
        "flat":        calib_dir / "flat.parquet",
        "dead":        calib_dir / "dead.parquet",
        "read":        calib_dir / "read.parquet",
        "linear_corr": calib_dir / "linear_corr.parquet",
    }

    # --- Vérification de l'existence de tous les fichiers
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers de calibration manquants pour planète {planet_id} "
            f"(instrument={instrument}) : {missing}\n"
            f"Dossier attendu : {calib_dir}"
        )

    # --- Chargement et conversion
    dark        = pd.read_parquet(paths["dark"]).values.astype(np.float64)
    flat        = pd.read_parquet(paths["flat"]).values.astype(np.float64)
    dead        = pd.read_parquet(paths["dead"]).values.astype(bool)
    read        = pd.read_parquet(paths["read"]).values.astype(np.float64)
    linear_corr = pd.read_parquet(paths["linear_corr"]).values.astype(np.float64)

    # --- Reshape selon l'instrument
    if instrument == "AIRS":
        dark        = dark.reshape(AIRS_SHAPE_SPATIAL, AIRS_SHAPE_SPECTRAL)
        flat        = flat.reshape(AIRS_SHAPE_SPATIAL, AIRS_SHAPE_SPECTRAL)
        dead        = dead.reshape(AIRS_SHAPE_SPATIAL, AIRS_SHAPE_SPECTRAL)
        read        = read.reshape(AIRS_SHAPE_SPATIAL, AIRS_SHAPE_SPECTRAL)
        linear_corr = linear_corr.reshape(
            N_LINEAR_CORR_COEFFS, AIRS_SHAPE_SPATIAL, AIRS_SHAPE_SPECTRAL
        )
        # --- Coupe spectrale : on garde uniquement les 283 longueurs d'onde utiles
        if cut_inf is not None and cut_sup is not None:
            dark        = dark[:, cut_inf:cut_sup]
            flat        = flat[:, cut_inf:cut_sup]
            dead        = dead[:, cut_inf:cut_sup]
            read        = read[:, cut_inf:cut_sup]
            linear_corr = linear_corr[:, :, cut_inf:cut_sup]

    elif instrument == "FGS1":
        # Détecteur photométrique carré — pas de dimension spectrale
        dark        = dark.reshape(FGS1_SHAPE, FGS1_SHAPE)
        flat        = flat.reshape(FGS1_SHAPE, FGS1_SHAPE)
        dead        = dead.reshape(FGS1_SHAPE, FGS1_SHAPE)
        read        = read.reshape(FGS1_SHAPE, FGS1_SHAPE)
        linear_corr = linear_corr.reshape(N_LINEAR_CORR_COEFFS, FGS1_SHAPE, FGS1_SHAPE)

    else:
        raise ValueError(
            f"Instrument inconnu : '{instrument}'. "
            f"Valeurs acceptées : 'AIRS', 'FGS1'."
        )

    logger.debug(
        f"load_calib | planète={planet_id} instrument={instrument} "
        f"dark.shape={dark.shape}"
    )
    return dark, flat, dead, read, linear_corr
