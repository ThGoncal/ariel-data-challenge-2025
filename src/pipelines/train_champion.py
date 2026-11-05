"""
train_champion.py — Entraîne le champion v0.2.0 sur les spectres reconstruits, le mesure, le versionne
======================================================================================================
Étape DVC `train_champion` (après `spectra`) :
  1. entraîne Ridge[Rp², astro] sur les planètes d'entraînement (data/processed/spectra_v2) ;
  2. score GLL sur la validation FIXE (+ intervalle bootstrap sur les planètes) ;
  3. validation croisée répétée sur TOUTES les planètes (la validation fixe de 20 planètes est trop bruitée
     pour trancher à elle seule) ;
  4. sauvegarde models/champion_v2/{bundle.joblib, meta.json} + reports/champion/metrics.json (dvc metrics) ;
  5. avec --mlflow : journalise params/métriques/artefacts ; avec --register NOM : enregistre le modèle dans le
     registre MLflow (alias `champion`).

    python -m src.pipelines.train_champion --params params.yaml                       # local, sans MLflow
    python -m src.pipelines.train_champion --params params.yaml --mlflow --register ariel-ridge-adaptive-depth
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.ml.features import build_target_matrix
from src.ml.metrics import evaluate_on_validation
from src.models.champion import feature_matrix, fit_champion, predict_champion, score_champion
from src.signal_processing.spectra import ExtractionParams

logger = logging.getLogger(__name__)
ROOT = Path("data/processed/spectra_v2")


def load_layout(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.read_csv(Path(root) / "val.csv")
    files = sorted((Path(root) / "batches").glob("batch_*.csv"))
    if not files or val.empty:
        raise SystemExit(f"Spectres absents ou vides dans {root} : lance d'abord `dvc repro spectra`.")
    train = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("planet_id")
    return train, val


def bootstrap_val(bundle: dict, val: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """Intervalle à 90 % du score de validation, par rééchantillonnage des planètes."""
    X, _ = feature_matrix(val, bundle["power"])
    y, _ = build_target_matrix(val)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    pred = predict_champion(bundle, val[ok])
    y, ids = y[ok], val["planet_id"].to_numpy()[ok]
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        s.append(float(evaluate_on_validation(y[i], pred.mu[i], pred.sigma[i], ids[i],
                                              bundle["naive_mean"], bundle["naive_sigma"])))
    return float(np.percentile(s, 5)), float(np.percentile(s, 95))


def cross_val(all_df: pd.DataFrame, alpha: float, power: int, n_splits: int, repeats: int, seed: int) -> np.ndarray:
    scores = []
    for r in range(repeats):
        for a, b in KFold(n_splits=min(n_splits, len(all_df)), shuffle=True, random_state=seed + r).split(all_df):
            bundle = fit_champion(all_df.iloc[a], alpha=alpha, power=power)
            scores.append(score_champion(bundle, all_df.iloc[b]))
    return np.array(scores)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def register_model(name: str, bundle_path: Path, meta: dict, src_dir: Path) -> str:
    """Enregistre le bundle comme modèle pyfunc dans le registre MLflow (alias `champion`) ; retourne la version."""
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient

    from src.models.champion_pyfunc import ChampionPyfunc

    kwargs = dict(python_model=ChampionPyfunc(), artifacts={"bundle": str(bundle_path)},
                  code_paths=[str(src_dir)], registered_model_name=name,
                  pip_requirements=["mlflow", "scikit-learn", "numpy", "pandas", "scipy", "joblib"])
    try:
        mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:                                    # anciennes versions de MLflow
        mlflow.pyfunc.log_model(artifact_path="model", **kwargs)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    version = str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(name, "champion", version)
    for k, v in meta.items():
        client.set_model_version_tag(name, version, k, str(v))
    return version


def run(params: dict, root: Path = ROOT, models_dir: Path = Path("models/champion_v2"),
        reports_dir: Path = Path("reports/champion"), use_mlflow: bool = False, register: str | None = None,
        cv: bool = True, seed: int = 42, src_dir: Path = Path("src")) -> dict:
    cp = {"alpha": 10.0, "power": 2, "cv_folds": 5, "cv_repeats": 3, **(params.get("champion") or {})}
    ext = ExtractionParams.from_params(params)
    train, val = load_layout(root)
    bundle = fit_champion(train, alpha=cp["alpha"], power=cp["power"])
    val_score = score_champion(bundle, val)
    ci05, ci95 = bootstrap_val(bundle, val, seed=seed)
    metrics = {"gll_val": val_score, "gll_val_ci05": ci05, "gll_val_ci95": ci95,
               "n_train": bundle["n_train"], "n_val": int(len(val))}
    if cv:
        s = cross_val(pd.concat([val, train], ignore_index=True), cp["alpha"], cp["power"],
                      cp["cv_folds"], cp["cv_repeats"], seed)
        metrics.update({"gll_cv_mean": float(s.mean()), "gll_cv_sd": float(s.std()), "cv_folds_total": int(len(s))})

    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    meta = {"git_commit": git_commit(), "curves_sha256": manifest.get("curves_sha256", "unknown"),
            **{f"extraction_{k}": v for k, v in ext.as_dict().items()}, "alpha": cp["alpha"], "power": cp["power"],
            "n_train": metrics["n_train"], "gll_val": round(val_score, 4)}
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "bundle.joblib")
    (models_dir / "meta.json").write_text(json.dumps({**meta, **metrics}, indent=2))
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if use_mlflow:
        import mlflow
        mlflow.set_experiment("ariel-champion")
        with mlflow.start_run(run_name="champion-v0.2.0-ridge-depth"):
            mlflow.log_params({"alpha": cp["alpha"], "power": cp["power"], **ext.as_dict()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.set_tags({k: str(v) for k, v in meta.items()})
            mlflow.log_artifact(str(models_dir / "bundle.joblib"), artifact_path="champion")
            mlflow.log_artifact(str(models_dir / "meta.json"), artifact_path="champion")
            if register:
                metrics["registered_version"] = register_model(register, models_dir / "bundle.joblib", meta, src_dir)
    return metrics


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--models-dir", default="models/champion_v2")
    ap.add_argument("--reports-dir", default="reports/champion")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--register", default=None, metavar="NOM", help="nom du modèle dans le registre MLflow")
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    params = yaml.safe_load(Path(a.params).read_text())
    m = run(params, Path(a.root), Path(a.models_dir), Path(a.reports_dir), a.mlflow, a.register, not a.no_cv,
            src_dir=Path(__file__).resolve().parents[1])
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
