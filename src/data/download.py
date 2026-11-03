"""
download.py — Ariel Data Challenge 2025
=======================================
Téléchargement des données via l'API Kaggle officielle.

Stratégie :
  - Téléchargement fichier par fichier (planète par planète pour train)
  - Reprise automatique via un fichier de log JSON
  - Compatible avec DVC (appelé depuis dvc.yaml)
  - Paramètres configurables via params.yaml ou CLI

Usage :
  # Télécharger uniquement les fichiers meta (CSV racine)
  python src/data/download.py --mode meta

  # Télécharger les N premières planètes train
  python src/data/download.py --mode train --n-planets 50

  # Télécharger toutes les planètes train
  python src/data/download.py --mode train

  # Télécharger les données test
  python src/data/download.py --mode test

  # Tout en une seule commande
  python src/data/download.py --mode all --n-planets 100
"""

# === Standard library
import os
import json
import time
import argparse
import logging
from pathlib import Path

# === Scientific stack
import pandas as pd
from tqdm import tqdm

# === Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    raise ImportError(
        "Le package 'kaggle' n'est pas installé. Lancez : pip install kaggle"
    )

# === Dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optionnel si kaggle.json est déjà en place

# =============================================================================
# CONFIGURATION
# =============================================================================

COMPETITION = "ariel-data-challenge-2025"

# Chemins relatifs à la racine du projet
RAW_DIR        = Path("data/raw")
LOG_DIR        = Path("data/raw/.download_logs")
META_FILES     = [
    "adc_info.csv",
    "wavelengths.csv",
    "axis_info.parquet",
    "train_star_info.csv",
    "test_star_info.csv",
    "train.csv",
    "sample_submission.csv",
]

# Fichiers par planète (AIRS + FGS1 + calibrations)
PLANET_FILES_TRAIN = [
    "AIRS-CH0_signal_0.parquet",
    "AIRS-CH0_calibration_0/dark.parquet",
    "AIRS-CH0_calibration_0/dead.parquet",
    "AIRS-CH0_calibration_0/flat.parquet",
    "AIRS-CH0_calibration_0/linear_corr.parquet",
    "AIRS-CH0_calibration_0/read.parquet",
    "FGS1_signal_0.parquet",
    "FGS1_calibration_0/dark.parquet",
    "FGS1_calibration_0/dead.parquet",
    "FGS1_calibration_0/flat.parquet",
    "FGS1_calibration_0/linear_corr.parquet",
    "FGS1_calibration_0/read.parquet",
]

# Retry
MAX_RETRIES    = 3
RETRY_DELAY_S  = 5

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("data/raw", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/raw/download.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# GESTIONNAIRE DE LOG (reprise)
# =============================================================================

class DownloadLog:
    """
    Gère la persistance des fichiers déjà téléchargés.
    Format : JSON  { "downloaded": ["train/100/signal.parquet", ...] }
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: set = self._load()

    def _load(self) -> set:
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                return set(json.load(f).get("downloaded", []))
        return set()

    def _save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump({"downloaded": sorted(self._data)}, f, indent=2)

    def is_done(self, key: str) -> bool:
        return key in self._data

    def mark_done(self, key: str):
        self._data.add(key)
        self._save()

    def __len__(self):
        return len(self._data)


# =============================================================================
# FONCTIONS DE TÉLÉCHARGEMENT
# =============================================================================

def get_api() -> KaggleApi:
    """Authentifie et retourne l'instance API Kaggle."""
    api = KaggleApi()
    api.authenticate()
    logger.info("✅ Authentification Kaggle OK")
    return api


def download_file_with_retry(
    api: KaggleApi,
    competition: str,
    file_path: str,
    dest_dir: Path,
    log: DownloadLog,
    log_key: str,
) -> bool:
    """
    Télécharge un fichier unique avec retry.
    Retourne True si succès, False sinon.

    Parameters
    ----------
    api          : instance KaggleApi authentifiée
    competition  : slug de la compétition Kaggle
    file_path    : chemin du fichier dans l'arborescence Kaggle
                   ex: "train/100/AIRS-CH0_signal_0.parquet"
    dest_dir     : dossier de destination local
    log          : DownloadLog pour la reprise
    log_key      : clé unique pour le log (généralement = file_path)
    """
    if log.is_done(log_key):
        return True  # Déjà téléchargé

    dest_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.competition_download_file(
                competition=competition,
                file_name=file_path,
                path=str(dest_dir),
                force=False,
                quiet=True,
            )
            log.mark_done(log_key)
            return True

        except Exception as exc:
            logger.warning(
                f"  ⚠️  Tentative {attempt}/{MAX_RETRIES} échouée pour '{file_path}': {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)

    logger.error(f"  ❌ Échec définitif : {file_path}")
    return False


# =============================================================================
# TÉLÉCHARGEMENT DES FICHIERS META (CSV racine)
# =============================================================================

def download_meta(api: KaggleApi, log: DownloadLog) -> dict:
    """Télécharge les fichiers CSV/parquet à la racine de la compétition."""
    logger.info("📥 Téléchargement des fichiers meta...")
    results = {"success": 0, "skip": 0, "fail": 0}

    for fname in tqdm(META_FILES, desc="Fichiers meta"):
        dest_dir = RAW_DIR
        log_key = f"meta/{fname}"

        if log.is_done(log_key):
            results["skip"] += 1
            continue

        ok = download_file_with_retry(api, COMPETITION, fname, dest_dir, log, log_key)
        if ok:
            results["success"] += 1
        else:
            results["fail"] += 1

    logger.info(
        f"  Meta → ✅ {results['success']} téléchargés | "
        f"⏭️ {results['skip']} déjà présents | "
        f"❌ {results['fail']} échecs"
    )
    return results


# =============================================================================
# TÉLÉCHARGEMENT DES PLANÈTES TRAIN
# =============================================================================

def download_train_planets(
    api: KaggleApi,
    log: DownloadLog,
    planet_ids: list,
    n_planets: int = None,
) -> dict:
    """
    Télécharge les données de chaque planète du set train,
    fichier par fichier pour permettre la reprise granulaire.

    Parameters
    ----------
    planet_ids : liste des planet_id (depuis train_star_info.csv)
    n_planets  : si défini, limite le téléchargement aux N premières planètes
    """
    if n_planets:
        planet_ids = planet_ids[:n_planets]
        logger.info(f"📥 Téléchargement TRAIN : {len(planet_ids)} planètes (limité à {n_planets})")
    else:
        logger.info(f"📥 Téléchargement TRAIN : {len(planet_ids)} planètes")

    results = {"success": 0, "skip": 0, "fail": 0}

    for planet_id in tqdm(planet_ids, desc="Planètes TRAIN", unit="planète"):
        planet_id_str = str(int(planet_id))

        for rel_file in PLANET_FILES_TRAIN:
            # Chemin Kaggle : "train/<planet_id>/AIRS-CH0_signal_0.parquet"
            kaggle_path = f"train/{planet_id_str}/{rel_file}"
            log_key = kaggle_path

            if log.is_done(log_key):
                results["skip"] += 1
                continue

            # Dossier local : data/raw/train/<planet_id>/AIRS-CH0_calibration_0/
            sub_dir = Path(rel_file).parent
            dest_dir = RAW_DIR / "train" / planet_id_str / sub_dir

            ok = download_file_with_retry(
                api, COMPETITION, kaggle_path, dest_dir, log, log_key
            )
            if ok:
                results["success"] += 1
            else:
                results["fail"] += 1

    logger.info(
        f"  Train → ✅ {results['success']} fichiers | "
        f"⏭️ {results['skip']} déjà présents | "
        f"❌ {results['fail']} échecs"
    )
    return results


# =============================================================================
# TÉLÉCHARGEMENT DES PLANÈTES TEST
# =============================================================================

def download_test_planets(
    api: KaggleApi,
    log: DownloadLog,
    planet_ids: list,
) -> dict:
    """Télécharge les données du set test."""
    logger.info(f"📥 Téléchargement TEST : {len(planet_ids)} planètes")
    results = {"success": 0, "skip": 0, "fail": 0}

    # Le test n'a pas de labels → pas de train.csv pour les targets
    # Mais la structure des dossiers est identique
    test_files = [
        "AIRS-CH0_signal_0.parquet",
        "AIRS-CH0_calibration_0/dark.parquet",
        "AIRS-CH0_calibration_0/dead.parquet",
        "AIRS-CH0_calibration_0/flat.parquet",
        "AIRS-CH0_calibration_0/linear_corr.parquet",
        "AIRS-CH0_calibration_0/read.parquet",
        "FGS1_signal_0.parquet",
        "FGS1_calibration_0/dark.parquet",
        "FGS1_calibration_0/dead.parquet",
        "FGS1_calibration_0/flat.parquet",
        "FGS1_calibration_0/linear_corr.parquet",
        "FGS1_calibration_0/read.parquet",
    ]

    for planet_id in tqdm(planet_ids, desc="Planètes TEST", unit="planète"):
        planet_id_str = str(int(planet_id))

        for rel_file in test_files:
            kaggle_path = f"test/{planet_id_str}/{rel_file}"
            log_key = kaggle_path

            if log.is_done(log_key):
                results["skip"] += 1
                continue

            sub_dir = Path(rel_file).parent
            dest_dir = RAW_DIR / "test" / planet_id_str / sub_dir

            ok = download_file_with_retry(
                api, COMPETITION, kaggle_path, dest_dir, log, log_key
            )
            if ok:
                results["success"] += 1
            else:
                results["fail"] += 1

    logger.info(
        f"  Test → ✅ {results['success']} fichiers | "
        f"⏭️ {results['skip']} déjà présents | "
        f"❌ {results['fail']} échecs"
    )
    return results


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(mode: str = "all", n_planets: int = None):
    """
    Point d'entrée principal.

    Parameters
    ----------
    mode       : "meta" | "train" | "test" | "all"
    n_planets  : nombre de planètes à télécharger (None = toutes)
    """
    # --- Création des dossiers
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Authentification
    api = get_api()

    # --- Log de reprise
    log = DownloadLog(LOG_DIR / "download_status.json")
    logger.info(f"📋 Fichiers déjà téléchargés (reprise) : {len(log)}")

    # --- Mode META (toujours fait en premier)
    if mode in ("meta", "all"):
        download_meta(api, log)

    # --- Lecture des planet_ids depuis les CSVs meta
    train_info_path = RAW_DIR / "train_star_info.csv"
    test_info_path  = RAW_DIR / "test_star_info.csv"

    # --- Mode TRAIN
    if mode in ("train", "all"):
        if not train_info_path.exists():
            logger.error(
                "❌ train_star_info.csv introuvable. "
                "Lance d'abord : python download.py --mode meta"
            )
            return

        df_train_info = pd.read_csv(train_info_path)
        planet_ids_train = df_train_info["planet_id"].tolist()
        download_train_planets(api, log, planet_ids_train, n_planets=n_planets)

    # --- Mode TEST
    if mode in ("test", "all"):
        if not test_info_path.exists():
            logger.error(
                "❌ test_star_info.csv introuvable. "
                "Lance d'abord : python download.py --mode meta"
            )
            return

        df_test_info = pd.read_csv(test_info_path)
        planet_ids_test = df_test_info["planet_id"].tolist()
        download_test_planets(api, log, planet_ids_test)

    logger.info("🎉 Téléchargement terminé !")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Téléchargement des données Ariel Data Challenge 2025 via Kaggle API"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["meta", "train", "test", "all"],
        help=(
            "Mode de téléchargement : "
            "'meta' (fichiers CSV racine), "
            "'train' (planètes train), "
            "'test' (planètes test), "
            "'all' (tout)"
        ),
    )
    parser.add_argument(
        "--n-planets",
        type=int,
        default=None,
        help="Nombre de planètes train à télécharger (défaut : toutes)",
    )

    args = parser.parse_args()
    main(mode=args.mode, n_planets=args.n_planets)
