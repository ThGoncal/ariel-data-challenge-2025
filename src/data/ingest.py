"""
src/data/ingest.py — Chargement des données — Ariel Data Challenge 2025
========================================================================
Extension du module template pour supporter les formats Ariel :
    - Parquets multi-niveaux (signal + calibrations)
    - CSV meta (star_info, train labels)
    - Validation de la structure des fichiers téléchargés
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# FONCTIONS DU TEMPLATE (conservées et étendues)
# =============================================================================

def load_raw_data(source_path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier brut (CSV, parquet, JSON) en DataFrame.
    Fonction générique du template — conservée telle quelle.
    """
    path = Path(source_path)
    logger.info(f"Chargement de {path}")

    loaders = {
        ".csv":     pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json":    pd.read_json,
    }
    loader = loaders.get(path.suffix)
    if not loader:
        raise ValueError(f"Format non supporté : {path.suffix}")

    df = loader(path)
    logger.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Sauvegarde un DataFrame traité en parquet.
    Fonction générique du template — conservée telle quelle.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Données sauvegardées : {path}")


# =============================================================================
# FONCTIONS SPÉCIFIQUES ARIEL
# =============================================================================

def load_meta_files(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Charge les 7 fichiers meta du dataset Ariel en un seul appel.

    Ces fichiers sont à la racine du dataset et contiennent :
        - adc_info.csv         : gain/offset ADC des détecteurs
        - wavelengths.csv      : 283 longueurs d'onde utiles (μm)
        - axis_info.parquet    : peigne spectral complet + temps d'intégration
        - train_star_info.csv  : paramètres stellaires/orbitaux (train)
        - test_star_info.csv   : paramètres stellaires/orbitaux (test)
        - train.csv            : labels — spectres de transmission vrais
        - sample_submission.csv: format de soumission Kaggle

    Parameters
    ----------
    raw_dir : chemin vers data/raw/

    Returns
    -------
    dict : clé = nom court, valeur = DataFrame
    """
    raw_dir = Path(raw_dir)

    meta_files = {
        "adc_info":          raw_dir / "adc_info.csv",
        "wavelengths":       raw_dir / "wavelengths.csv",
        "axis_info":         raw_dir / "axis_info.parquet",
        "train_star_info":   raw_dir / "train_star_info.csv",
        "test_star_info":    raw_dir / "test_star_info.csv",
        "train_labels":      raw_dir / "train.csv",
        "sample_submission": raw_dir / "sample_submission.csv",
    }

    # Vérification de présence avant chargement
    missing = [name for name, path in meta_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers meta manquants : {missing}\n"
            f"Lance d'abord : python src/data/collect.py --mode meta"
        )

    loaded = {}
    for name, path in meta_files.items():
        loaded[name] = load_raw_data(path)

    logger.info(f"✅ {len(loaded)} fichiers meta chargés depuis {raw_dir}")
    return loaded


def validate_planet_files(planet_dir: str | Path, instrument: str = "AIRS") -> bool:
    """
    Vérifie qu'un dossier planète contient tous les fichiers attendus.

    Utilisé par collect.py avant de déclencher le processing —
    évite de lancer un traitement sur des données incomplètes.

    Parameters
    ----------
    planet_dir : chemin vers data/raw/train/<planet_id>/
    instrument : "AIRS" ou "FGS1"

    Returns
    -------
    bool : True si tous les fichiers sont présents
    """
    planet_dir = Path(planet_dir)

    required = {
        "AIRS": [
            "AIRS-CH0_signal_0.parquet",
            "AIRS-CH0_calibration_0/dark.parquet",
            "AIRS-CH0_calibration_0/flat.parquet",
            "AIRS-CH0_calibration_0/dead.parquet",
            "AIRS-CH0_calibration_0/read.parquet",
            "AIRS-CH0_calibration_0/linear_corr.parquet",
        ],
        "FGS1": [
            "FGS1_signal_0.parquet",
            "FGS1_calibration_0/dark.parquet",
            "FGS1_calibration_0/flat.parquet",
            "FGS1_calibration_0/dead.parquet",
            "FGS1_calibration_0/read.parquet",
            "FGS1_calibration_0/linear_corr.parquet",
        ],
    }

    files_to_check = required.get("AIRS", []) + required.get("FGS1", [])
    missing = [f for f in files_to_check if not (planet_dir / f).exists()]

    if missing:
        logger.warning(f"Planète {planet_dir.name} — fichiers manquants : {missing}")
        return False

    return True


def load_accumulated_spectra(processed_dir: str | Path) -> pd.DataFrame | None:
    """
    Charge le CSV accumulé train_with_spectra.csv.

    Retourne None si le fichier n'existe pas encore
    (aucun batch n'a encore été traité).

    Parameters
    ----------
    processed_dir : chemin vers data/processed/

    Returns
    -------
    DataFrame ou None
    """
    path = Path(processed_dir) / "train_with_spectra.csv"

    if not path.exists():
        logger.info("train_with_spectra.csv inexistant — aucun batch traité")
        return None

    df = pd.read_csv(path)
    logger.info(f"CSV accumulé chargé : {len(df)} planètes")
    return df


def get_batch_planet_ids(
    raw_dir:    str | Path,
    split:      str,
    batch_idx:  int,
    batch_size: int,
) -> list[str]:
    """
    Retourne les planet_id du batch demandé depuis le fichier star_info.

    Parameters
    ----------
    raw_dir    : chemin vers data/raw/
    split      : "train" ou "test"
    batch_idx  : indice du batch (0-indexé)
    batch_size : nombre de planètes par batch

    Returns
    -------
    Liste de planet_id sous forme de chaînes
    """
    info_path = Path(raw_dir) / f"{split}_star_info.csv"
    df        = pd.read_csv(info_path)
    ids       = [str(int(pid)) for pid in sorted(df["planet_id"].tolist())]

    start = batch_idx * batch_size
    end   = min(start + batch_size, len(ids))

    if start >= len(ids):
        logger.warning(f"Batch {batch_idx} hors limites ({len(ids)} planètes)")
        return []

    batch = ids[start:end]
    logger.info(
        f"Batch {batch_idx} : planètes {start}..{end-1} "
        f"({len(batch)} planètes | split={split})"
    )
    return batch
