"""
scripts/run_pipeline.py — Pipeline local complet — Ariel Data Challenge 2025
=============================================================================
Lance le pipeline incrémental complet en dehors d'Airflow.
Utile pour le développement local, les tests et le débogage.

Usage :
    # Pipeline complet (toutes les planètes)
    python scripts/run_pipeline.py

    # Test rapide : 2 batchs de 10 planètes
    python scripts/run_pipeline.py --batch-size 10 --max-batches 2

    # Reprendre un pipeline interrompu
    python scripts/run_pipeline.py --resume

    # Sans MLflow (développement)
    python scripts/run_pipeline.py --no-mlflow --batch-size 10 --max-batches 1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ÉTAPES DU PIPELINE
# =============================================================================

def step_download_meta():
    """Étape 1 : téléchargement des fichiers meta."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 1 — Téléchargement des fichiers meta")
    logger.info("=" * 60)

    from src.data.collect import get_api, download_meta
    api = get_api()

    meta_files_present = all(
        Path(f"data/raw/{f}").exists()
        for f in ["adc_info.csv", "wavelengths.csv", "train_star_info.csv"]
    )
    if meta_files_present:
        logger.info("Fichiers meta déjà présents → skip")
        return True

    success = download_meta(api)
    if not success:
        logger.error("❌ Échec du téléchargement des fichiers meta")
        return False

    logger.info("✅ Fichiers meta téléchargés")
    return True


def step_incremental_loop(
    batch_size:  int,
    max_batches: int,
    min_delta:   float,
    patience:    int,
    use_mlflow:  bool,
    model_type:  str,
):
    """
    Étape 2 : boucle incrémentale principale.

    Pour chaque batch :
        1. Télécharge les planètes
        2. Traite le signal → Rp/Rs → append CSV
        3. Entraîne le modèle sur le CSV accumulé
        4. Évalue le score GLL
        5. Décide de continuer ou d'arrêter
    """
    from src.data.collect import (
        get_api, get_planet_ids, download_planet,
        process_batch, cleanup_batch, BatchState,
    )
    from src.data.ingest import load_accumulated_spectra
    from src.monitoring.drift import monitor_spectral_drift, monitor_prediction_quality, push_metrics_to_prometheus

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    api        = get_api()
    planet_ids = get_planet_ids("train")
    n_planets  = len(planet_ids)
    n_batches  = min(max_batches, (n_planets + batch_size - 1) // batch_size)

    # État de la boucle incrémentale
    best_score     = -1.0
    best_run_id    = None
    no_improve     = 0
    score_history  = []

    logger.info(f"Boucle incrémentale | {n_batches} batchs × {batch_size} planètes")
    logger.info(f"Critère d'arrêt : patience={patience} | min_delta={min_delta}")

    for batch_idx in range(n_batches):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"BATCH {batch_idx}/{n_batches-1}")
        logger.info("=" * 60)
        t_start = time.perf_counter()

        # --- 2a. Téléchargement
        start  = batch_idx * batch_size
        end    = min(start + batch_size, n_planets)
        batch  = planet_ids[start:end]

        logger.info(f"Téléchargement de {len(batch)} planètes ({start}..{end-1})...")
        failed = []
        for pid in batch:
            ok = download_planet(api, pid, "train")
            if not ok:
                failed.append(pid)
        success = [p for p in batch if p not in failed]

        if not success:
            logger.error(f"Batch {batch_idx} : 0 planètes téléchargées → skip")
            continue

        # --- 2b. Traitement signal → append CSV
        logger.info(f"Traitement de {len(success)} planètes...")
        process_batch(success, "train")
        cleanup_batch(success, "train")

        # --- 2c. Chargement du CSV accumulé
        df_train = load_accumulated_spectra("data/processed")
        if df_train is None or len(df_train) < 10:
            logger.warning("Pas assez de données pour entraîner → skip")
            continue

        # Charger test si disponible
        test_path = Path("data/processed/test_with_spectra.csv")
        df_test   = None
        if test_path.exists():
            import pandas as pd
            df_test = pd.read_csv(test_path)

        # --- 2d. Entraînement + évaluation
        logger.info(f"Entraînement sur {len(df_train)} planètes accumulées...")

        run_name = f"batch_{batch_idx:03d}_{len(df_train)}planets"

        if not use_mlflow:
            # Mode sans MLflow : entraînement simplifié
            val_score, run_id = _train_without_mlflow(df_train, params, model_type)
        elif model_type == "dl" and df_test is not None:
            from src.models.train import train_ariel_dl
            val_score, run_id = train_ariel_dl(df_train, df_test, params, run_name, len(df_train))
        elif df_test is not None:
            from src.models.train import train_ariel_baseline
            val_score, run_id = train_ariel_baseline(df_train, df_test, params, run_name, len(df_train))
        else:
            logger.warning("test_with_spectra.csv absent → entraînement sans soumission")
            val_score, run_id = _train_without_mlflow(df_train, params, model_type)

        # --- 2e. Monitoring drift
        drift_result = monitor_spectral_drift(batch_idx)
        push_metrics_to_prometheus(batch_idx, val_score, len(df_train), drift_result["drift_score"])

        # --- 2f. Décision : continuer ou arrêter ?
        elapsed = time.perf_counter() - t_start
        score_history.append({
            "batch":     batch_idx,
            "score":     val_score,
            "n_planets": len(df_train),
            "run_id":    run_id,
            "elapsed_s": elapsed,
        })

        improved = val_score > best_score + min_delta

        if improved:
            best_score  = val_score
            best_run_id = run_id
            no_improve  = 0
            logger.info(f"✅ Amélioration : {val_score:.6f} (+{val_score - (best_score - (val_score - best_score)):.6f})")
        else:
            no_improve += 1
            logger.info(f"📉 Pas d'amélioration ({no_improve}/{patience}) | meilleur : {best_score:.6f}")

        logger.info(
            f"Score : {val_score:.6f} | "
            f"Planètes : {len(df_train)} | "
            f"Durée : {elapsed:.0f}s"
        )

        if no_improve >= patience:
            logger.info(f"🛑 Arrêt automatique après {patience} batchs sans amélioration")
            break

    # --- Rapport final
    monitor_prediction_quality(score_history)
    return best_score, best_run_id, score_history


def _train_without_mlflow(df_train, params, model_type):
    """Entraînement simplifié sans MLflow (développement local)."""
    import numpy as np
    from sklearn.linear_model import MultiTaskLassoCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from src.features.build_features import prepare_train_data
    from src.models.evaluate import evaluate_predictions
    from src.ml.features import compute_naive_stats

    bl = params["baseline"]

    X, y, _, _ = prepare_train_data(df_train, drop_eccentricity=True)

    X_train, X_val, y_train, y_val, _, idx_val = train_test_split(
        X, y, np.arange(len(df_train)), test_size=0.2, random_state=42
    )
    pids_val = df_train["planet_id"].values[idx_val]

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_tr_sc  = scaler_X.transform(X_train)
    X_val_sc = scaler_X.transform(X_val)
    y_tr_sc  = scaler_y.transform(y_train)

    alphas = np.logspace(bl["alphas_min"], bl["alphas_max"], bl["n_alphas"])
    model  = MultiTaskLassoCV(alphas=alphas, cv=min(bl["cv"], len(X_tr_sc)), max_iter=bl["max_iter"], n_jobs=-1)
    model.fit(X_tr_sc, y_tr_sc)

    preds_mean = scaler_y.inverse_transform(model.predict(X_val_sc))
    preds_std  = np.full_like(preds_mean, 1e-3)

    naive_mean, naive_sigma = compute_naive_stats(y_train)
    val_score = evaluate_predictions(y_val, preds_mean, preds_std, pids_val, naive_mean, naive_sigma)

    return val_score, "local_run"


def step_generate_submission(best_run_id: str):
    """Étape 3 : génération de la soumission Kaggle finale."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 3 — Génération de la soumission Kaggle")
    logger.info("=" * 60)

    sub_path = Path("models/submission_incremental.csv")
    if sub_path.exists():
        logger.info(f"Soumission : {sub_path}")
    else:
        logger.warning("Soumission non trouvée — utilise la dernière soumission disponible")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline local Ariel Data Challenge 2025")
    parser.add_argument("--batch-size",  type=int,   default=50,    help="Planètes par batch")
    parser.add_argument("--max-batches", type=int,   default=22,    help="Nombre max de batchs")
    parser.add_argument("--min-delta",   type=float, default=0.005, help="Amélioration minimale requise")
    parser.add_argument("--patience",    type=int,   default=3,     help="Patience avant arrêt")
    parser.add_argument("--model",       type=str,   default="baseline", choices=["baseline", "dl"])
    parser.add_argument("--no-mlflow",   action="store_true", help="Désactive MLflow")
    parser.add_argument("--resume",      action="store_true", help="Reprend depuis l'état sauvegardé")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PIPELINE INCRÉMENTAL — ARIEL DATA CHALLENGE 2025")
    logger.info(f"batch_size={args.batch_size} | max_batches={args.max_batches}")
    logger.info(f"patience={args.patience} | min_delta={args.min_delta}")
    logger.info(f"model={args.model} | mlflow={not args.no_mlflow}")
    logger.info("=" * 60)

    # Étape 1
    if not step_download_meta():
        sys.exit(1)

    # Étape 2 — boucle incrémentale
    best_score, best_run_id, history = step_incremental_loop(
        batch_size  = args.batch_size,
        max_batches = args.max_batches,
        min_delta   = args.min_delta,
        patience    = args.patience,
        use_mlflow  = not args.no_mlflow,
        model_type  = args.model,
    )

    # Étape 3
    step_generate_submission(best_run_id)

    # Résumé
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info(f"  Meilleur score GLL : {best_score:.6f}")
    logger.info(f"  Batchs exécutés    : {len(history)}")
    logger.info(f"  Planètes traitées  : {history[-1]['n_planets'] if history else 0}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
