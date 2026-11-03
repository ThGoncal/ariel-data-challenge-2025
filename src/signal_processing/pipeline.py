"""
pipeline.py — Orchestration du traitement signal Ariel Data Challenge 2025
===========================================================================
Responsabilité : assembler les modules io, calibration et transit pour
traiter une planète complète, puis paralléliser sur toutes les planètes.

Flux de données pour une planète :
    fichiers parquet
        ↓  io.load_signal / io.load_calib
    ndarrays bruts
        ↓  calibration.ADC_convert
    signal en unités de flux
        ↓  CDS (Correlated Double Sampling)
    signal différentiel normalisé par dt
        ↓  calibration.calibrate_and_clean_cube  (dark, flat, dead, hot pixels)
    signal calibré
        ↓  calibration.apply_linear_corr
    signal corrigé non-linéarité
        ↓  calibration.bin_time
    signal binné
        ↓  transit.calcul_Rp_median  (AIRS)
        ↓  transit.calcul_Rp_fgs1   (FGS1)
    vecteur Rp/Rs  (1 FGS1 + 282 AIRS = 283 features)
"""

# === Standard library
import logging
import time
from pathlib import Path

# === Scientific stack
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# === Modules internes
from src.signal_processing.io import load_signal, load_calib
from src.signal_processing.calibration import (
    ADC_convert,
    calibrate_and_clean_cube,
    apply_linear_corr,
    bin_time,
    ensure_ndarray,
)
from src.signal_processing.transit import calcul_Rp_median, calcul_Rp_fgs1

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES — BINNING PAR DÉFAUT
# =============================================================================

# Ces valeurs ont été déterminées empiriquement pour le challenge 2025.
# AIRS : 5 frames consécutives moyennées → bon compromis bruit/résolution
# FGS1 : 20 frames (cadence plus rapide, plus de bruit de lecture par frame)
DEFAULT_BINNING_AIRS = 5
DEFAULT_BINNING_FGS1 = 20


# =============================================================================
# PIPELINE D'UNE SEULE PLANÈTE
# =============================================================================

def process_one_planet(
    planet_id: int | str,
    base_path: str | Path,
    base_folder: str,
    df_axis_info: pd.DataFrame,
    gain: float,
    offset: float,
    cut_inf: int,
    cut_sup: int,
    binning_airs: int = DEFAULT_BINNING_AIRS,
    binning_fgs1: int = DEFAULT_BINNING_FGS1,
) -> dict | None:
    """
    Traite une planète complète et retourne son vecteur de features.

    Cette fonction est conçue pour être appelée en parallèle par joblib.
    Elle est autonome : elle ne partage pas de mémoire avec les autres workers.

    En cas d'erreur (fichier manquant, fit qui diverge…), elle retourne None
    plutôt que de faire planter tout le batch — le filtrage est fait en aval
    dans build_spectra_dataframe.

    Parameters
    ----------
    planet_id    : identifiant de la planète (int ou str, ex: 100)
    base_path    : racine des données (ex: Path("data/raw"))
    base_folder  : "train" ou "test"
    df_axis_info : DataFrame chargé depuis axis_info.parquet
                   (contient les temps d'intégration AIRS)
    gain, offset : paramètres ADC (depuis adc_info.csv)
    cut_inf, cut_sup : bornes de la coupe spectrale AIRS (depuis wavelength_index)
    binning_airs, binning_fgs1 : facteurs de binning temporel

    Returns
    -------
    dict avec clés : "planet_id", "Rp_1" (FGS1), "Rp_2".."Rp_283" (AIRS)
    ou None si erreur
    """
    planet_id_str = str(int(planet_id))
    logger.info(f"▶ Traitement planète {planet_id_str}")

    try:
        # ===================================================================
        # AIRS — Spectromètre infrarouge
        # ===================================================================

        # --- 1. Chargement signal brut AIRS
        signal_raw_airs = load_signal(
            planet_id=planet_id_str,
            instrument="AIRS",
            base_path=base_path,
            base_folder=base_folder,
            signal_filename="AIRS-CH0_signal_0.parquet",
        )

        # --- 2. Chargement calibrations AIRS
        dark_a, flat_a, dead_a, read_a, lincorr_a = load_calib(
            planet_id=planet_id_str,
            base_path=base_path,
            base_folder=base_folder,
            calibration_dir="AIRS-CH0_calibration_0",
            instrument="AIRS",
            cut_inf=cut_inf,
            cut_sup=cut_sup,
        )

        # --- 3. Temps d'intégration AIRS (depuis axis_info.parquet)
        dt_airs = df_axis_info["AIRS-CH0-integration_time"].dropna().values

        # --- 4. Conversion ADC inverse
        n_time_airs = signal_raw_airs.shape[0]
        signal_airs  = ADC_convert(signal_raw_airs, gain, offset)

        # --- 5. Reshape (n_time, n_pixels) → (n_time, 32, 356)
        #         puis coupe spectrale → (n_time, 32, n_wl_utiles)
        signal_airs = signal_airs.reshape(n_time_airs, 32, 356)
        signal_airs = signal_airs[:, :, cut_inf:cut_sup]
        signal_airs = ensure_ndarray(signal_airs)

        # --- 6. Correlated Double Sampling (CDS)
        #    On soustrait chaque frame "start" à la frame "end" suivante.
        #    Les frames sont interleaved : [start_0, end_0, start_1, end_1, ...]
        #    On ajoute +0.1s aux temps "end" pour corriger l'offset de lecture.
        signal_cds_airs = signal_airs[1::2, :, :] - signal_airs[::2, :, :]
        dt_airs_corr    = dt_airs.copy()
        dt_airs_corr[1::2] += 0.1
        dt_airs_corr    = dt_airs_corr[1::2]       # on garde seulement les "end"

        # Normalisation par temps d'intégration → unités de flux par seconde
        signal_cds_airs = signal_cds_airs / dt_airs_corr[:, None, None]

        # --- 7. Calibration complète (dark, flat, dead, hot pixels)
        signal_cds_airs = calibrate_and_clean_cube(
            signal_cds_airs,
            dark_a, flat_a, dead_a, read_a,
            dt_airs_corr,
        )

        # --- 8. Correction non-linéarité
        signal_cds_airs = apply_linear_corr(signal_cds_airs, lincorr_a)

        # --- 9. Binning temporel
        signal_cds_airs = bin_time(signal_cds_airs, binning_airs)

        # --- 10. Extraction Rp/Rs (méthode médiane, parallèle Numba)
        Rp_airs = calcul_Rp_median(signal_cds_airs)   # (n_wl,) ≈ 282 valeurs

        # ===================================================================
        # FGS1 — Photomètre large bande
        # ===================================================================

        # --- 1. Chargement signal brut FGS1
        signal_raw_fgs1 = load_signal(
            planet_id=planet_id_str,
            instrument="FGS1",
            base_path=base_path,
            base_folder=base_folder,
            signal_filename="FGS1_signal_0.parquet",
        )

        # --- 2. Chargement calibrations FGS1
        dark_f, flat_f, dead_f, read_f, lincorr_f = load_calib(
            planet_id=planet_id_str,
            base_path=base_path,
            base_folder=base_folder,
            calibration_dir="FGS1_calibration_0",
            instrument="FGS1",
        )

        # --- 3. Temps d'intégration FGS1 (fixe : 0.1s par frame)
        n_time_fgs1 = signal_raw_fgs1.shape[0]
        dt_fgs1     = np.ones(n_time_fgs1) * 0.1

        # --- 4. Conversion ADC + reshape → (n_time, 32, 32)
        signal_fgs1 = ADC_convert(signal_raw_fgs1, gain, offset)
        signal_fgs1 = signal_fgs1.reshape(n_time_fgs1, 32, 32)
        signal_fgs1 = ensure_ndarray(signal_fgs1)

        # --- 5. CDS
        signal_cds_fgs1 = signal_fgs1[1::2, :, :] - signal_fgs1[::2, :, :]
        dt_fgs1_corr    = dt_fgs1[1::2]
        signal_cds_fgs1 = signal_cds_fgs1 / dt_fgs1_corr[:, None, None]

        # --- 6. Calibration
        signal_cds_fgs1 = calibrate_and_clean_cube(
            signal_cds_fgs1,
            dark_f, flat_f, dead_f, read_f,
            dt_fgs1_corr,
        )

        # --- 7. Correction non-linéarité
        signal_cds_fgs1 = apply_linear_corr(signal_cds_fgs1, lincorr_f)

        # --- 8. Binning
        signal_cds_fgs1 = bin_time(signal_cds_fgs1, binning_fgs1)

        # --- 9. Extraction Rp (photométrique, fit super-gaussienne)
        Rp_fgs1 = calcul_Rp_fgs1(signal_cds_fgs1)     # (1,)

        # ===================================================================
        # CONSTRUCTION DU VECTEUR DE FEATURES
        # ===================================================================
        # Convention Ariel : Rp_1 = FGS1 (photométrique), Rp_2..283 = AIRS
        row = {"planet_id": int(planet_id)}
        row["Rp_1"] = float(Rp_fgs1[0]) if Rp_fgs1.size == 1 else np.nan

        for j, val in enumerate(Rp_airs, start=2):
            row[f"Rp_{j}"] = float(val)

        logger.info(f"✅ Planète {planet_id_str} traitée — {len(Rp_airs)} longueurs d'onde AIRS")
        return row

    except FileNotFoundError as exc:
        logger.warning(f"⚠️  Planète {planet_id_str} ignorée (fichier manquant) : {exc}")
        return None
    except Exception as exc:
        logger.error(f"❌ Erreur planète {planet_id_str} : {repr(exc)}")
        return None


# =============================================================================
# TRAITEMENT EN BATCH — TOUTES LES PLANÈTES
# =============================================================================

def build_spectra_dataframe(
    base_path: str | Path,
    base_folder: str,
    axis_info_path: str | Path,
    adc_info: pd.DataFrame,
    star_info: pd.DataFrame,
    cut_inf: int,
    cut_sup: int,
    binning_airs: int = DEFAULT_BINNING_AIRS,
    binning_fgs1: int = DEFAULT_BINNING_FGS1,
    n_jobs: int = 4,
    planet_ids: list = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame des spectres pour toutes les planètes.

    Parallélise le traitement sur `n_jobs` workers via joblib (threading).
    Le backend "threading" est choisi plutôt que "loky" (multiprocessing)
    car les fonctions Numba libèrent le GIL — on évite ainsi le coût
    de la sérialisation des données entre processus.

    Parameters
    ----------
    base_path     : racine des données
    base_folder   : "train" ou "test"
    axis_info_path: chemin vers axis_info.parquet
    adc_info      : DataFrame depuis adc_info.csv (gain/offset)
    star_info     : DataFrame depuis train_star_info.csv ou test_star_info.csv
    cut_inf, cut_sup : bornes coupe spectrale AIRS
    binning_airs, binning_fgs1 : facteurs de binning
    n_jobs        : nombre de workers parallèles
    planet_ids    : liste de planet_id à traiter (None = tous depuis star_info)

    Returns
    -------
    df_spectra : DataFrame
        Colonnes : planet_id, Rp_1 (FGS1), Rp_2..Rp_283 (AIRS)
        Une ligne par planète traitée avec succès.
    """
    t0 = time.perf_counter()

    # --- Paramètres ADC (communs à toutes les planètes)
    gain   = float(adc_info.loc[0, "AIRS-CH0_adc_gain"])
    offset = float(adc_info.loc[0, "AIRS-CH0_adc_offset"])

    # --- Chargement axis_info (temps d'intégration AIRS)
    df_axis_info = pd.read_parquet(axis_info_path)

    # --- Liste des planètes à traiter
    if planet_ids is None:
        planet_ids = star_info["planet_id"].tolist()

    logger.info(
        f"build_spectra_dataframe | {base_folder} | "
        f"{len(planet_ids)} planètes | {n_jobs} workers"
    )

    # --- Parallélisation avec joblib (threading pour compatibilité Numba)
    rows = Parallel(
        n_jobs=n_jobs,
        verbose=5,
        backend="threading",   # Numba libère le GIL → threading efficace
    )(
        delayed(process_one_planet)(
            planet_id=pid,
            base_path=base_path,
            base_folder=base_folder,
            df_axis_info=df_axis_info,
            gain=gain,
            offset=offset,
            cut_inf=cut_inf,
            cut_sup=cut_sup,
            binning_airs=binning_airs,
            binning_fgs1=binning_fgs1,
        )
        for pid in tqdm(planet_ids, desc=f"Planètes {base_folder}", unit="planète")
    )

    # --- Filtrage des erreurs (None)
    rows_clean = [r for r in rows if r is not None]
    n_errors   = len(rows) - len(rows_clean)

    # --- Bilan de performance
    elapsed    = time.perf_counter() - t0
    n_ok       = len(rows_clean)
    mean_time  = elapsed / n_ok if n_ok > 0 else float("nan")

    logger.info("=" * 60)
    logger.info(f"✅ {n_ok}/{len(rows)} planètes traitées avec succès")
    logger.info(f"❌ {n_errors} erreurs")
    logger.info(f"⏱  Temps total   : {elapsed / 60:.1f} min")
    logger.info(f"⏱  Temps moyen   : {mean_time:.2f} s/planète")
    logger.info(f"⏱  Estimation 1100 planètes : {mean_time * 1100 / 3600:.2f} h")
    logger.info("=" * 60)

    return pd.DataFrame(rows_clean)
