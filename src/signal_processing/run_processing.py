"""
run_processing.py — Point d'entrée du pipeline de traitement du signal
=======================================================================
Ce script est appelé par DVC (dvc.yaml) pour produire les fichiers
train_with_spectra.csv et test_with_spectra.csv.

Usage :
    # Traiter le set train complet
    python src/signal_processing/run_processing.py --split train

    # Traiter 10 planètes pour tester
    python src/signal_processing/run_processing.py --split train --n-planets 10

    # Traiter le set test
    python src/signal_processing/run_processing.py --split test

    # Les deux en une commande
    python src/signal_processing/run_processing.py --split all
"""

import os
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.signal_processing.io import wavelength_index
from src.signal_processing.pipeline import build_spectra_dataframe

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/processed/processing.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CHARGEMENT DES PARAMÈTRES (params.yaml)
# =============================================================================

def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# MAIN
# =============================================================================

def main(split: str = "all", n_planets: int = None):
    params    = load_params()
    sp        = params["signal"]
    data_cfg  = params["data"]

    RAW_DIR       = Path(data_cfg["raw_dir"])
    PROCESSED_DIR = Path(data_cfg["processed_dir"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Indices de coupe spectrale (communs train/test)
    wl_idx   = wavelength_index(
        wavelengths_path=RAW_DIR / "wavelengths.csv",
        axis_info_path=RAW_DIR / "axis_info.parquet",
    )
    cut_inf, cut_sup = int(wl_idx.min()), int(wl_idx.max()) + 1

    # --- Infos ADC
    adc_info = pd.read_csv(RAW_DIR / "adc_info.csv")

    # =========================================================================
    # TRAITEMENT TRAIN
    # =========================================================================
    if split in ("train", "all"):
        logger.info("=" * 60)
        logger.info("TRAITEMENT TRAIN")
        logger.info("=" * 60)

        star_info  = pd.read_csv(RAW_DIR / "train_star_info.csv")
        planet_ids = star_info["planet_id"].tolist()
        if n_planets:
            planet_ids = planet_ids[:n_planets]

        df_spectra = build_spectra_dataframe(
            base_path=RAW_DIR,
            base_folder="train",
            axis_info_path=RAW_DIR / "axis_info.parquet",
            adc_info=adc_info,
            star_info=star_info,
            cut_inf=cut_inf,
            cut_sup=cut_sup,
            binning_airs=sp["binning_airs"],
            binning_fgs1=sp["binning_fgs1"],
            n_jobs=sp["n_jobs"],
            planet_ids=planet_ids,
        )

        # Merge avec train.csv (labels) et star_info (features stellaires)
        df_train_labels = pd.read_csv(RAW_DIR / "train.csv")
        df_final = (
            df_train_labels
            .merge(df_spectra,  on="planet_id", how="left")
            .merge(star_info,   on="planet_id", how="left")
        )

        out_path = PROCESSED_DIR / "train_with_spectra.csv"
        df_final.to_csv(out_path, index=False)
        logger.info(f"✅ Sauvegardé : {out_path}  ({len(df_final)} lignes)")

    # =========================================================================
    # TRAITEMENT TEST
    # =========================================================================
    if split in ("test", "all"):
        logger.info("=" * 60)
        logger.info("TRAITEMENT TEST")
        logger.info("=" * 60)

        star_info_test = pd.read_csv(RAW_DIR / "test_star_info.csv")

        df_spectra_test = build_spectra_dataframe(
            base_path=RAW_DIR,
            base_folder="test",
            axis_info_path=RAW_DIR / "axis_info.parquet",
            adc_info=adc_info,
            star_info=star_info_test,
            cut_inf=cut_inf,
            cut_sup=cut_sup,
            binning_airs=sp["binning_airs"],
            binning_fgs1=sp["binning_fgs1"],
            n_jobs=sp["n_jobs"],
        )

        df_test_final = df_spectra_test.merge(star_info_test, on="planet_id", how="left")

        out_path = PROCESSED_DIR / "test_with_spectra.csv"
        df_test_final.to_csv(out_path, index=False)
        logger.info(f"✅ Sauvegardé : {out_path}  ({len(df_test_final)} lignes)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement du signal Ariel 2025"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--n-planets",
        type=int,
        default=None,
        help="Nombre de planètes à traiter (défaut : toutes)",
    )
    args = parser.parse_args()
    main(split=args.split, n_planets=args.n_planets)
