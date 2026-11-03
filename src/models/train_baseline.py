"""
train_baseline.py — Modèle baseline Ariel Data Challenge 2025
=============================================================
Entraîne un MultiTaskLassoCV avec bootstrap pour estimer l'incertitude,
intègre la métrique officielle score() et loggue tout dans MLflow.

Corrections apportées vs ML_04.py :
    - BUG 1 : X_kaggle_test séparé de X_val (plus d'écrasement)
    - BUG 2 : scaler fitté sur X_train uniquement (pas de data leakage)
              inverse_transform appliqué aux prédictions avant évaluation
    - BUG 3 : bootstrap agrégé en (mean, std) → soumission complète générée
    - BUG 4 : features construites par nom explicite (plus de pop() fragile)
    - BUG 5 : score() ne mute plus ses arguments (corrigé dans metrics.py)

Usage :
    python src/models/train_baseline.py
    python src/models/train_baseline.py --no-mlflow   # sans tracking MLflow
"""

# === Standard library
import argparse
import json
import logging
import os
from pathlib import Path

# === Scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.utils import resample
from tqdm import tqdm

# === MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# === Dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === Modules internes
from src.ml.features import build_feature_matrix, build_target_matrix, compute_naive_stats
from src.ml.metrics  import evaluate_on_validation, build_submission_dataframe

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/processed/train_baseline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CHARGEMENT DES PARAMÈTRES
# =============================================================================

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# ENTRAÎNEMENT BOOTSTRAP
# =============================================================================

def train_bootstrap(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    n_bootstrap: int,
    alphas_grid: np.ndarray,
    cv: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entraîne N modèles sur des échantillons bootstrap et agrège les prédictions.

    Le bootstrap sert à estimer l'incertitude de prédiction :
    chaque modèle est entraîné sur un rééchantillonnage avec remise du train set.
    La variabilité des prédictions entre les N modèles donne une estimation
    de l'écart-type σ_pred requis par la métrique GLL.

    CORRECTION BUG 3 : les prédictions sont agrégées ici en (mean, std),
    et retournées en espace normalisé — l'inversion scaler est faite dans main().

    Parameters
    ----------
    X_train_scaled : (n_train, n_features) — features normalisées
    y_train_scaled : (n_train, 283) — targets normalisées
    X_val_scaled   : (n_val, n_features) — features validation normalisées
    n_bootstrap    : nombre d'itérations bootstrap
    alphas_grid    : grille de valeurs alpha pour la CV
    cv             : nombre de folds de cross-validation
    max_iter       : nombre max d'itérations du solveur

    Returns
    -------
    preds_mean_scaled : (n_val, 283) — moyenne des prédictions bootstrap
    preds_std_scaled  : (n_val, 283) — écart-type des prédictions bootstrap
    """
    all_preds = []   # liste de n_bootstrap arrays (n_val, 283)

    for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Rééchantillonnage avec remise
        X_boot, y_boot = resample(X_train_scaled, y_train_scaled, random_state=i)

        model = MultiTaskLassoCV(
            alphas=alphas_grid,
            cv=cv,
            max_iter=max_iter,
            random_state=i,
            n_jobs=-1,        # utilise tous les cœurs pour la CV interne
        )
        model.fit(X_boot, y_boot)
        all_preds.append(model.predict(X_val_scaled))   # (n_val, 283)

    # Stack en 3D : (n_bootstrap, n_val, 283)
    preds_array = np.stack(all_preds, axis=0)

    # Agrégation : mean et std sur l'axe bootstrap
    preds_mean_scaled = preds_array.mean(axis=0)   # (n_val, 283)
    preds_std_scaled  = preds_array.std(axis=0)    # (n_val, 283)

    return preds_mean_scaled, preds_std_scaled


# =============================================================================
# MAIN
# =============================================================================

def main(use_mlflow: bool = True):
    params   = load_params()
    bl_cfg   = params["baseline"]
    pre_cfg  = params["preprocess"]
    data_cfg = params["data"]

    PROCESSED_DIR = Path(data_cfg["processed_dir"])
    MODELS_DIR    = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. CHARGEMENT DES DONNÉES
    # =========================================================================
    logger.info("Chargement des données...")
    df_train = pd.read_csv(PROCESSED_DIR / "train_with_spectra.csv")
    df_test  = pd.read_csv(PROCESSED_DIR / "test_with_spectra.csv")

    # =========================================================================
    # 2. FEATURE ENGINEERING (CORRECTION BUG 4 : pas de pop() fragile)
    # =========================================================================
    logger.info("Feature engineering...")
    X_all,       feat_cols = build_feature_matrix(df_train, drop_eccentricity=pre_cfg["drop_eccentricity"])
    y_all,       wl_cols   = build_target_matrix(df_train)
    X_kaggle,    _         = build_feature_matrix(df_test,  drop_eccentricity=pre_cfg["drop_eccentricity"])

    # =========================================================================
    # 3. SPLIT TRAIN / VALIDATION (CORRECTION BUG 1 : X_kaggle protégé)
    # =========================================================================
    # On nomme clairement X_val et y_val pour ne pas confondre avec X_kaggle
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_all, y_all, np.arange(len(df_train)),
        test_size=pre_cfg["test_size"],
        random_state=pre_cfg["random_state"],
    )
    planet_ids_val = df_train["planet_id"].values[idx_val]

    logger.info(f"Split : {len(X_train)} train | {len(X_val)} val | {len(X_kaggle)} test Kaggle")

    # =========================================================================
    # 4. STANDARDISATION (CORRECTION BUG 2 : fit sur X_train uniquement)
    # =========================================================================
    # RÈGLE FONDAMENTALE : le scaler est fitté UNIQUEMENT sur X_train.
    # Le fitter sur X_all entier causerait du "data leakage" : les stats
    # de X_val influenceraient la normalisation, biaisant l'évaluation.
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled  = scaler_X.transform(X_train)
    X_val_scaled    = scaler_X.transform(X_val)
    X_kaggle_scaled = scaler_X.transform(X_kaggle)   # jamais utilisé pour le fit !

    y_train_scaled  = scaler_y.transform(y_train)

    # Baseline naïve (sur y_train uniquement, pas y_all)
    naive_mean, naive_sigma = compute_naive_stats(y_train)

    # =========================================================================
    # 5. ENTRAÎNEMENT BOOTSTRAP
    # =========================================================================
    alphas_grid = np.logspace(bl_cfg["alphas_min"], bl_cfg["alphas_max"], bl_cfg["n_alphas"])

    logger.info(f"Entraînement : {bl_cfg['n_bootstrap']} itérations bootstrap...")
    preds_mean_scaled, preds_std_scaled = train_bootstrap(
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_val_scaled=X_val_scaled,
        n_bootstrap=bl_cfg["n_bootstrap"],
        alphas_grid=alphas_grid,
        cv=bl_cfg["cv"],
        max_iter=bl_cfg["max_iter"],
    )

    # =========================================================================
    # 6. INVERSION DU SCALER (CORRECTION BUG 2 suite)
    # =========================================================================
    # Les prédictions sont en espace normalisé → on repasse en espace original
    # pour pouvoir appeler score() qui travaille en valeurs physiques
    y_val_pred_mean = scaler_y.inverse_transform(preds_mean_scaled)    # (n_val, 283)

    # Pour l'écart-type : on ne peut pas utiliser inverse_transform directement
    # car il inclut le décalage (mean). On utilise uniquement le facteur d'échelle.
    # σ_original = σ_normalisé × std_scaler
    y_val_pred_std = preds_std_scaled * scaler_y.scale_               # (n_val, 283)

    # Clip : les std doivent être positives
    y_val_pred_std = np.clip(y_val_pred_std, a_min=1e-9, a_max=None)

    # =========================================================================
    # 7. ÉVALUATION AVEC LA MÉTRIQUE OFFICIELLE (CORRECTION BUG 3)
    # =========================================================================
    val_score = evaluate_on_validation(
        y_val_true=y_val,
        y_pred_mean=y_val_pred_mean,
        y_pred_std=y_val_pred_std,
        planet_ids_val=planet_ids_val,
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
    )
    logger.info(f"Score GLL validation : {val_score:.6f}")

    # Métriques supplémentaires
    mse_val = float(np.mean((y_val - y_val_pred_mean) ** 2))
    logger.info(f"MSE validation : {mse_val:.8f}")

    metrics = {
        "val_gll_score": val_score,
        "val_mse":       mse_val,
        "naive_mean":    naive_mean,
        "naive_sigma":   naive_sigma,
    }

    # Sauvegarde des métriques pour DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # =========================================================================
    # 8. GÉNÉRATION DE LA SOUMISSION KAGGLE (CORRECTION BUG 3 suite)
    # =========================================================================
    logger.info("Génération des prédictions Kaggle...")

    # On ré-entraîne un modèle final sur tout X_all pour la soumission
    # (pas sur X_train seulement — on utilise toutes les données disponibles)
    X_all_scaled = scaler_X.transform(X_all)
    y_all_scaled = scaler_y.transform(y_all)

    # Bootstrap final sur X_all pour la soumission
    preds_kaggle_scaled_list = []
    for i in tqdm(range(bl_cfg["n_bootstrap"]), desc="Bootstrap Kaggle"):
        X_boot, y_boot = resample(X_all_scaled, y_all_scaled, random_state=i + 1000)
        model = MultiTaskLassoCV(
            alphas=alphas_grid,
            cv=bl_cfg["cv"],
            max_iter=bl_cfg["max_iter"],
            random_state=i,
            n_jobs=-1,
        )
        model.fit(X_boot, y_boot)
        preds_kaggle_scaled_list.append(model.predict(X_kaggle_scaled))

    preds_kaggle_array = np.stack(preds_kaggle_scaled_list, axis=0)
    kaggle_mean_scaled = preds_kaggle_array.mean(axis=0)
    kaggle_std_scaled  = preds_kaggle_array.std(axis=0)

    kaggle_mean = scaler_y.inverse_transform(kaggle_mean_scaled)
    kaggle_std  = np.clip(kaggle_std_scaled * scaler_y.scale_, a_min=1e-9, a_max=None)

    planet_ids_kaggle = df_test["planet_id"].values
    df_submission = build_submission_dataframe(planet_ids_kaggle, kaggle_mean, kaggle_std)

    submission_path = MODELS_DIR / "submission_baseline.csv"
    df_submission.to_csv(submission_path, index=False)
    logger.info(f"Soumission sauvegardée : {submission_path}")

    # =========================================================================
    # 9. TRACKING MLFLOW
    # =========================================================================
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment("ariel-2025-baseline")

        with mlflow.start_run(run_name="MultiTaskLassoCV_bootstrap"):
            # Paramètres
            mlflow.log_params({
                "model":         "MultiTaskLassoCV",
                "n_bootstrap":   bl_cfg["n_bootstrap"],
                "alphas_min":    bl_cfg["alphas_min"],
                "alphas_max":    bl_cfg["alphas_max"],
                "n_alphas":      bl_cfg["n_alphas"],
                "cv":            bl_cfg["cv"],
                "max_iter":      bl_cfg["max_iter"],
                "test_size":     pre_cfg["test_size"],
                "n_features":    len(feat_cols),
                "n_train":       len(X_train),
                "n_val":         len(X_val),
            })

            # Métriques
            mlflow.log_metrics(metrics)

            # Artefacts
            mlflow.log_artifact("metrics.json")
            mlflow.log_artifact(str(submission_path))

            # Graphe de corrélation Rp vs wl
            _log_correlation_plot(df_train, feat_cols, wl_cols)

            logger.info("✅ Run MLflow loggué")
    else:
        logger.info("MLflow non utilisé (--no-mlflow ou package absent)")

    logger.info("=" * 60)
    logger.info(f"Score GLL final sur validation : {val_score:.6f}")
    logger.info("=" * 60)
    return val_score


# =============================================================================
# UTILITAIRES
# =============================================================================

def _log_correlation_plot(df_train: pd.DataFrame, feat_cols: list, wl_cols: list):
    """Génère et loggue le graphe de corrélation Rp_i vs wl_i dans MLflow."""
    rp_cols = [c for c in feat_cols if c.startswith("Rp_")]
    correlations = [
        np.corrcoef(df_train[rp_cols[i]], df_train[wl_cols[i]])[0, 1]
        for i in range(min(len(rp_cols), len(wl_cols)))
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(1, len(correlations) + 1), correlations, marker="o", markersize=3)
    ax.set_title("Corrélation Rp_i vs wl_i par longueur d'onde")
    ax.set_xlabel("Index de longueur d'onde")
    ax.set_ylabel("Corrélation de Pearson")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    mlflow.log_figure(fig, "correlation_Rp_vs_wl.png")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement baseline Ariel 2025")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Désactive le tracking MLflow",
    )
    args = parser.parse_args()
    main(use_mlflow=not args.no_mlflow)
