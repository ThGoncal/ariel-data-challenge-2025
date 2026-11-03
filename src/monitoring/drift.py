"""
src/monitoring/drift.py — Monitoring drift — Ariel Data Challenge 2025
=======================================================================
Étend le module drift du template pour les données spectrales Ariel.
Surveille deux types de drift :
    1. Drift des features (Rp_i) entre batchs successifs
    2. Drift des prédictions (wl_i) entre itérations du pipeline incrémental
"""

from pathlib import Path

import pandas as pd
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently non installé — monitoring drift désactivé")


# =============================================================================
# FONCTION DU TEMPLATE (étendue)
# =============================================================================

def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data:   pd.DataFrame,
    output_path:    str | Path = "monitoring/evidently/reports/drift_report.html",
) -> bool:
    """
    Génère un rapport de drift Evidently entre deux datasets.

    Compatible avec l'interface du template + étendue pour Ariel.

    Parameters
    ----------
    reference_data : DataFrame de référence (train complet ou batch précédent)
    current_data   : DataFrame courant (nouveau batch ou prédictions)
    output_path    : chemin du rapport HTML

    Returns
    -------
    bool : True si drift détecté
    """
    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently non disponible → rapport non généré")
        return False

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference_data, current_data=current_data)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))

    # Extraction du résultat de drift global
    results       = report.as_dict()
    drift_detected = results["metrics"][2]["result"]["dataset_drift"]

    status = "⚠️  Drift détecté" if drift_detected else "✅ Pas de drift"
    logger.info(f"{status} | rapport : {output}")

    return drift_detected


# =============================================================================
# FONCTIONS SPÉCIFIQUES ARIEL
# =============================================================================

def monitor_spectral_drift(
    batch_idx:      int,
    processed_dir:  str | Path = "data/processed",
    output_dir:     str | Path = "monitoring/evidently/reports",
    n_wl_sample:    int        = 20,
) -> dict:
    """
    Surveille le drift spectral entre le batch courant et la référence.

    Sélectionne un échantillon de N longueurs d'onde (pour la performance)
    et génère un rapport Evidently comparant leur distribution.

    Le drift spectral peut indiquer :
        - Un problème dans le traitement du signal d'un batch
        - Des planètes structurellement différentes (biais de sélection)
        - Une erreur de calibration sur certaines planètes

    Parameters
    ----------
    batch_idx     : indice du batch courant (pour nommer le rapport)
    processed_dir : dossier contenant train_with_spectra.csv
    output_dir    : dossier de sortie des rapports HTML
    n_wl_sample   : nombre de longueurs d'onde à analyser (défaut: 20/283)

    Returns
    -------
    dict : {"drift_detected": bool, "drift_score": float, "report_path": str}
    """
    if not EVIDENTLY_AVAILABLE:
        return {"drift_detected": False, "drift_score": 0.0, "report_path": None}

    csv_path = Path(processed_dir) / "train_with_spectra.csv"
    if not csv_path.exists():
        logger.warning("train_with_spectra.csv introuvable → monitoring skip")
        return {"drift_detected": False, "drift_score": 0.0, "report_path": None}

    df = pd.read_csv(csv_path)

    # Sélection des colonnes Rp (features spectrales)
    rp_cols = [c for c in df.columns if c.startswith("Rp_")]
    if not rp_cols:
        logger.warning("Aucune colonne Rp_ trouvée → monitoring skip")
        return {"drift_detected": False, "drift_score": 0.0, "report_path": None}

    # Échantillonnage des longueurs d'onde (performances)
    import numpy as np
    sample_cols = rp_cols[::max(1, len(rp_cols) // n_wl_sample)][:n_wl_sample]

    # Référence = première moitié du CSV / Courant = deuxième moitié
    # (simule la comparaison batch précédent vs batch courant)
    n    = len(df)
    half = max(n // 2, 1)

    reference = df.iloc[:half][sample_cols]
    current   = df.iloc[half:][sample_cols]

    if len(current) == 0:
        logger.info("Pas assez de données pour le monitoring → skip")
        return {"drift_detected": False, "drift_score": 0.0, "report_path": None}

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current)

    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"spectral_drift_batch_{batch_idx:03d}.html"
    report.save_html(str(report_path))

    results       = report.as_dict()
    drift_detected = results["metrics"][1]["result"]["dataset_drift"]
    drift_score   = results["metrics"][1]["result"].get("drift_share", 0.0)

    logger.info(
        f"Drift spectral batch {batch_idx} | "
        f"détecté={drift_detected} | score={drift_score:.3f}"
    )

    return {
        "drift_detected": drift_detected,
        "drift_score":    drift_score,
        "report_path":    str(report_path),
    }


def monitor_prediction_quality(
    score_history: list[dict],
    output_path:   str | Path = "monitoring/evidently/reports/prediction_quality.html",
) -> None:
    """
    Génère un rapport d'évolution de la qualité des prédictions.

    Visualise comment le score GLL évolue à chaque batch.
    Exporté en HTML pour Evidently UI et en JSON pour MLflow.

    Parameters
    ----------
    score_history : [{"batch": int, "score": float, "n_planets": int}, ...]
    output_path   : chemin du rapport HTML
    """
    if not score_history:
        logger.warning("Historique vide → rapport qualité non généré")
        return

    df = pd.DataFrame(score_history)
    df = df.rename(columns={"score": "val_gll"})
    df["delta"]       = df["val_gll"].diff().fillna(0)
    df["cumul_improvement"] = (df["val_gll"] - df["val_gll"].iloc[0]).clip(lower=0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde JSON (compatible MLflow artifacts)
    json_path = output.with_suffix(".json")
    df.to_json(json_path, orient="records", indent=2)

    logger.info(
        f"Qualité prédictions | {len(df)} batchs | "
        f"score initial={df['val_gll'].iloc[0]:.4f} | "
        f"score final={df['val_gll'].iloc[-1]:.4f} | "
        f"meilleur={df['val_gll'].max():.4f} (batch {df['val_gll'].idxmax()})"
    )
    logger.info(f"  → rapport JSON : {json_path}")


def push_metrics_to_prometheus(
    batch_idx:     int,
    val_gll_score: float,
    n_planets:     int,
    drift_score:   float = 0.0,
) -> None:
    """
    Pousse les métriques vers Prometheus pour Grafana.

    Métriques exposées :
        - ariel_val_gll_score : score GLL courant
        - ariel_n_planets     : nombre de planètes traitées
        - ariel_drift_score   : score de drift spectral
        - ariel_batch_idx     : indice du batch courant
    """
    try:
        from src.monitoring.metrics import (
            MODEL_ACCURACY, DATA_DRIFT_SCORE
        )
        MODEL_ACCURACY.labels(
            model_name="ariel_baseline", version=f"batch_{batch_idx}"
        ).set(val_gll_score)

        DATA_DRIFT_SCORE.labels(feature="spectral_rp").set(drift_score)

        logger.info(
            f"Métriques Prometheus mises à jour | "
            f"gll={val_gll_score:.4f} | drift={drift_score:.4f}"
        )
    except Exception as e:
        logger.warning(f"Prometheus non disponible : {e}")
