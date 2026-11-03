"""
collect.py — Téléchargement Kaggle par batch — Ariel Data Challenge 2025
=========================================================================
Stratégie batch pour contrainte disque :
    - Télécharge N planètes à la fois (batch_size configurable)
    - Traite immédiatement chaque batch après téléchargement
    - Supprime les fichiers raw après traitement (libère le disque)
    - Reprend automatiquement là où il s'est arrêté (fichier d'état JSON)

Usage CLI :
    # Télécharger uniquement les fichiers meta (racine du dataset)
    python src/data/collect.py --mode meta

    # Télécharger + traiter par batch de 50 planètes
    python src/data/collect.py --mode batch --batch-size 50

    # Reprendre un batch interrompu
    python src/data/collect.py --mode batch --batch-size 50 --resume

    # Télécharger un batch spécifique (debug)
    python src/data/collect.py --mode batch --batch-index 3 --batch-size 50
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

COMPETITION    = "ariel-data-challenge-2025"
RAW_DIR        = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
STATE_FILE     = RAW_DIR / "batch_state.json"

# Fichiers à la racine du dataset (légers, versionnés par DVC)
META_FILES = [
    "adc_info.csv",
    "axis_info.parquet",
    "sample_submission.csv",
    "test_star_info.csv",
    "train.csv",
    "train_star_info.csv",
    "wavelengths.csv",
]

# Fichiers par planète (train)
# ~240 Mo par planète × 1100 = 264 Go total
PLANET_FILES = [
    "AIRS-CH0_signal_0.parquet",
    "FGS1_signal_0.parquet",
    # Calibrations AIRS
    "AIRS-CH0_calibration_0/dark.parquet",
    "AIRS-CH0_calibration_0/flat.parquet",
    "AIRS-CH0_calibration_0/dead.parquet",
    "AIRS-CH0_calibration_0/read.parquet",
    "AIRS-CH0_calibration_0/linear_corr.parquet",
    # Calibrations FGS1
    "FGS1_calibration_0/dark.parquet",
    "FGS1_calibration_0/flat.parquet",
    "FGS1_calibration_0/dead.parquet",
    "FGS1_calibration_0/read.parquet",
    "FGS1_calibration_0/linear_corr.parquet",
]

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RAW_DIR / "collect.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# GESTION DE L'ÉTAT (reprise après interruption)
# =============================================================================

class BatchState:
    """
    Persiste l'état d'avancement du téléchargement batch dans un fichier JSON.

    Structure du fichier batch_state.json :
    {
        "completed_batches": [0, 1, 2],      ← indices des batchs terminés
        "failed_planets": ["123", "456"],     ← planètes en erreur
        "total_planets": 1100,
        "batch_size": 50,
        "last_updated": "2025-03-01T10:30:00"
    }
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self._state     = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "completed_batches": [],
            "failed_planets":    [],
            "total_planets":     0,
            "batch_size":        50,
        }

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def is_batch_done(self, batch_idx: int) -> bool:
        return batch_idx in self._state["completed_batches"]

    def mark_batch_done(self, batch_idx: int):
        if batch_idx not in self._state["completed_batches"]:
            self._state["completed_batches"].append(batch_idx)
        self._save()
        logger.info(f"Batch {batch_idx} marqué comme terminé")

    def mark_planet_failed(self, planet_id: str):
        if planet_id not in self._state["failed_planets"]:
            self._state["failed_planets"].append(planet_id)
        self._save()

    def set_config(self, total_planets: int, batch_size: int):
        self._state["total_planets"] = total_planets
        self._state["batch_size"]    = batch_size
        self._save()

    @property
    def completed_batches(self) -> list:
        return self._state["completed_batches"]

    @property
    def failed_planets(self) -> list:
        return self._state["failed_planets"]

    def summary(self) -> str:
        n_done  = len(self._state["completed_batches"])
        n_total = (self._state["total_planets"] // self._state["batch_size"]) + 1
        return f"{n_done}/{n_total} batchs terminés | {len(self._state['failed_planets'])} planètes en erreur"


# =============================================================================
# API KAGGLE
# =============================================================================

def get_api() -> KaggleApi:
    """Initialise et authentifie l'API Kaggle."""
    api = KaggleApi()
    api.authenticate()
    logger.info("API Kaggle authentifiée")
    return api


def download_file_with_retry(
    api: KaggleApi,
    filename: str,
    dest_dir: Path,
    max_retries: int = 3,
    wait_between: float = 5.0,
) -> bool:
    """
    Télécharge un fichier Kaggle avec retry et backoff linéaire.

    Returns True si succès, False si échec après tous les retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            api.competition_download_file(
                COMPETITION,
                filename,
                str(dest_dir),
            )
            return True
        except Exception as e:
            wait = wait_between * attempt
            logger.warning(
                f"Tentative {attempt}/{max_retries} échouée pour {filename} : {e} "
                f"→ attente {wait}s"
            )
            if attempt < max_retries:
                time.sleep(wait)

    logger.error(f"Échec définitif pour {filename} après {max_retries} tentatives")
    return False


# =============================================================================
# MODE META : fichiers racine du dataset
# =============================================================================

def download_meta(api: KaggleApi):
    """
    Télécharge uniquement les 7 fichiers à la racine du dataset.
    Ces fichiers sont légers (<10 Mo total) et versionnés par DVC.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Téléchargement des {len(META_FILES)} fichiers meta...")

    ok, fail = 0, 0
    for filename in META_FILES:
        logger.info(f"  → {filename}")
        success = download_file_with_retry(api, filename, RAW_DIR)
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(2)   # pause pour éviter le rate limiting

    logger.info(f"Meta : {ok} succès, {fail} échecs")
    return fail == 0


# =============================================================================
# MODE BATCH : planètes par paquet
# =============================================================================

def get_planet_ids(split: str = "train") -> list[str]:
    """
    Lit les planet_id depuis le fichier star_info correspondant.
    Retourne une liste de chaînes de caractères triées.
    """
    info_file = RAW_DIR / f"{split}_star_info.csv"
    if not info_file.exists():
        raise FileNotFoundError(
            f"{info_file} introuvable. Lance d'abord : collect.py --mode meta"
        )
    df = pd.read_csv(info_file)
    return [str(int(pid)) for pid in sorted(df["planet_id"].tolist())]


def download_planet(
    api: KaggleApi,
    planet_id: str,
    split: str = "train",
) -> bool:
    """
    Télécharge tous les fichiers d'une planète dans data/raw/{split}/{planet_id}/.

    Structure cible :
        data/raw/train/123/
            AIRS-CH0_signal_0.parquet
            FGS1_signal_0.parquet
            AIRS-CH0_calibration_0/
                dark.parquet  flat.parquet  dead.parquet  read.parquet  linear_corr.parquet
            FGS1_calibration_0/
                dark.parquet  flat.parquet  dead.parquet  read.parquet  linear_corr.parquet
    """
    planet_dir = RAW_DIR / split / planet_id
    planet_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for rel_path in PLANET_FILES:
        # Chemin Kaggle : train/123/AIRS-CH0_signal_0.parquet
        kaggle_path = f"{split}/{planet_id}/{rel_path}"

        # Dossier de destination local (crée les sous-dossiers si nécessaire)
        local_subdir = planet_dir / Path(rel_path).parent
        local_subdir.mkdir(parents=True, exist_ok=True)

        success = download_file_with_retry(api, kaggle_path, local_subdir)
        if not success:
            ok = False
            logger.warning(f"Fichier manquant : planète {planet_id} / {rel_path}")

        time.sleep(0.5)   # pause légère entre fichiers

    return ok


def process_batch(planet_ids_batch: list[str], split: str = "train"):
    """
    Déclenche le traitement du signal pour un batch de planètes.

    Appelle run_processing.py avec la liste explicite des planet_ids
    plutôt que de tout retraiter — seules les planètes du batch courant
    sont traitées.

    Le résultat (lignes du CSV) est appendé à train_with_spectra.csv.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.signal_processing.io       import wavelength_index
    from src.signal_processing.pipeline import build_spectra_dataframe
    import yaml

    logger.info(f"Traitement de {len(planet_ids_batch)} planètes...")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    sp  = params["signal"]
    cfg = params["data"]

    # Index de coupe spectrale
    wl_idx   = wavelength_index(
        RAW_DIR / "wavelengths.csv",
        RAW_DIR / "axis_info.parquet",
    )
    cut_inf, cut_sup = int(wl_idx.min()), int(wl_idx.max()) + 1

    adc_info  = pd.read_csv(RAW_DIR / "adc_info.csv")
    star_info = pd.read_csv(RAW_DIR / f"{split}_star_info.csv")

    df_spectra = build_spectra_dataframe(
        base_path    = RAW_DIR,
        base_folder  = split,
        axis_info_path = RAW_DIR / "axis_info.parquet",
        adc_info     = adc_info,
        star_info    = star_info,
        cut_inf      = cut_inf,
        cut_sup      = cut_sup,
        binning_airs = sp["binning_airs"],
        binning_fgs1 = sp["binning_fgs1"],
        n_jobs       = sp["n_jobs"],
        planet_ids   = planet_ids_batch,
    )

    # Merge avec les labels (train uniquement)
    if split == "train":
        df_labels = pd.read_csv(RAW_DIR / "train.csv")
        df_stars  = pd.read_csv(RAW_DIR / "train_star_info.csv")
        df_out    = df_spectra.merge(df_labels, on="planet_id", how="left") \
                              .merge(df_stars,  on="planet_id", how="left")
    else:
        df_stars = pd.read_csv(RAW_DIR / "test_star_info.csv")
        df_out   = df_spectra.merge(df_stars, on="planet_id", how="left")

    # Append au CSV accumulateur (crée le fichier avec header si premier batch)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{split}_with_spectra.csv"
    header   = not out_path.exists()

    df_out.to_csv(out_path, mode="a", header=header, index=False)
    logger.info(f"  {len(df_out)} lignes ajoutées → {out_path}")


def cleanup_batch(planet_ids_batch: list[str], split: str = "train"):
    """
    Supprime les fichiers raw d'un batch après traitement réussi.
    Libère ~12 Go par batch de 50 planètes.
    """
    freed = 0
    for planet_id in planet_ids_batch:
        planet_dir = RAW_DIR / split / planet_id
        if planet_dir.exists():
            # Calcul de la taille avant suppression
            size = sum(f.stat().st_size for f in planet_dir.rglob("*") if f.is_file())
            shutil.rmtree(planet_dir)
            freed += size

    logger.info(f"Nettoyage : {freed / 1e9:.2f} Go libérés ({len(planet_ids_batch)} planètes)")


def run_batch_pipeline(
    batch_size: int  = 50,
    batch_index: int = None,
    resume: bool     = True,
    split: str       = "train",
    keep_raw: bool   = False,
):
    """
    Orchestre le pipeline batch complet :
        Pour chaque batch :
            1. Vérifie si déjà traité (reprise)
            2. Télécharge les planètes du batch
            3. Traite le signal → extrait Rp/Rs → append au CSV
            4. Supprime les fichiers raw (libère le disque)
            5. Marque le batch comme terminé dans l'état JSON

    Parameters
    ----------
    batch_size  : nombre de planètes par batch (défaut 50, ~12 Go)
    batch_index : si spécifié, traite uniquement ce batch (debug)
    resume      : si True, saute les batchs déjà terminés
    split       : "train" ou "test"
    keep_raw    : si True, ne supprime pas les fichiers raw après traitement
    """
    api   = get_api()
    state = BatchState()

    planet_ids = get_planet_ids(split)
    n_planets  = len(planet_ids)
    n_batches  = (n_planets + batch_size - 1) // batch_size

    state.set_config(n_planets, batch_size)

    logger.info("=" * 60)
    logger.info(f"Pipeline batch | split={split} | {n_planets} planètes | {n_batches} batchs")
    logger.info(f"Taille batch   | {batch_size} planètes (~{batch_size * 0.24:.0f} Go)")
    logger.info(f"Reprise        | {state.summary()}")
    logger.info("=" * 60)

    # Si batch_index spécifié → traite uniquement ce batch
    batch_range = [batch_index] if batch_index is not None else range(n_batches)

    for b_idx in batch_range:

        # --- Reprise : skip les batchs déjà terminés
        if resume and state.is_batch_done(b_idx):
            logger.info(f"Batch {b_idx}/{n_batches-1} : déjà traité, sauté")
            continue

        # --- Sélection des planètes du batch
        start = b_idx * batch_size
        end   = min(start + batch_size, n_planets)
        batch_planets = planet_ids[start:end]

        logger.info(f"\nBatch {b_idx}/{n_batches-1} : planètes {start}..{end-1} ({len(batch_planets)} planètes)")

        # --- Téléchargement
        failed_in_batch = []
        for planet_id in batch_planets:
            logger.info(f"  ↓ Planète {planet_id}")
            ok = download_planet(api, planet_id, split)
            if not ok:
                failed_in_batch.append(planet_id)
                state.mark_planet_failed(planet_id)

        # Planètes effectivement téléchargées
        success_planets = [p for p in batch_planets if p not in failed_in_batch]

        if not success_planets:
            logger.error(f"Batch {b_idx} : aucune planète téléchargée → skip")
            continue

        # --- Traitement du signal
        try:
            process_batch(success_planets, split)
        except Exception as e:
            logger.error(f"Batch {b_idx} : erreur de traitement → {e}")
            continue

        # --- Nettoyage des fichiers raw (libération disque)
        if not keep_raw:
            cleanup_batch(success_planets, split)

        # --- Marquer le batch comme terminé
        state.mark_batch_done(b_idx)
        logger.info(f"✅ Batch {b_idx} terminé | {state.summary()}")

    logger.info("=" * 60)
    logger.info(f"Pipeline terminé | {state.summary()}")
    logger.info(f"Fichier produit  : {PROCESSED_DIR}/{split}_with_spectra.csv")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement Kaggle par batch — Ariel Data Challenge 2025"
    )
    parser.add_argument(
        "--mode",
        choices=["meta", "batch"],
        required=True,
        help="meta = fichiers racine uniquement | batch = pipeline complet par batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Nombre de planètes par batch (défaut: 50, ~12 Go par batch)",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="Traite uniquement ce batch (debug). None = tous les batchs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reprend automatiquement là où le pipeline s'est arrêté (défaut: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Recommence depuis le début (ignore l'état JSON)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset à traiter (défaut: train)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        default=False,
        help="Conserve les fichiers raw après traitement (désactive la libération disque)",
    )

    args = parser.parse_args()

    if args.mode == "meta":
        api = get_api()
        download_meta(api)
    elif args.mode == "batch":
        run_batch_pipeline(
            batch_size  = args.batch_size,
            batch_index = args.batch_index,
            resume      = args.resume,
            split       = args.split,
            keep_raw    = args.keep_raw,
        )


if __name__ == "__main__":
    main()
