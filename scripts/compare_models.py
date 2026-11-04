"""
compare_models.py — Compare Ridge (v0.1.0), échelon A et échelon B HORS boucle incrémentale
===========================================================================================
Réutilise les spectres déjà traités (data/processed/incremental/val.csv + batches/*.csv) :
aucun téléchargement. Même validation fixe que la boucle, mêmes planètes d'entraînement, dans
le même ordre : les n premières planètes de l'ordre mélangé, comme au batch correspondant.

    uv run python scripts/compare_models.py                       # tailles 50, 100, toutes
    uv run python scripts/compare_models.py --sizes 30 60 90 140
    uv run python scripts/compare_models.py --mlflow              # journalise dans MLflow

Sortie : tableau (score GLL, intervalle bootstrap sur les planètes de validation, probabilité
d'être meilleur que le Ridge) + reports/compare/compare.csv.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features import build_feature_matrix, build_target_matrix, compute_naive_stats
from src.ml.metrics import evaluate_on_validation
from src.models.hetero import MODELS


def load_spectra(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.read_csv(root / "val.csv")
    files = sorted((root / "batches").glob("batch_*.csv"))
    if not files:
        raise SystemExit(f"Aucun batch dans {root / 'batches'} (as-tu fait `dvc pull` ?)")
    train = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("planet_id")
    return train, val


def to_arrays(df: pd.DataFrame):
    X, _ = build_feature_matrix(df)
    y, _ = build_target_matrix(df)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    return X[ok], y[ok], df["planet_id"].to_numpy()[ok]


def gll(y, pred, ids, nm, ns) -> float:
    return float(evaluate_on_validation(y, pred.mu, pred.sigma, ids, nm, ns))


def bootstrap_scores(y, pred, ids, nm, ns, idx_sets) -> np.ndarray:
    out = []
    for idx in idx_sets:
        out.append(float(evaluate_on_validation(y[idx], pred.mu[idx], pred.sigma[idx], ids[idx], nm, ns)))
    return np.array(out)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/incremental")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="nombres de planètes d'entraînement")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/compare")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()

    train_df, val_df = load_spectra(Path(a.root))
    Xva, yva, ids_va = to_arrays(val_df)
    n_all = len(train_df)
    sizes = sorted({min(s, n_all) for s in (a.sizes or [50, 100, n_all])})
    print(f"{n_all} planètes d'entraînement disponibles, {len(Xva)} en validation. Tailles : {sizes}")

    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, len(Xva), len(Xva)) for _ in range(a.n_boot)]
    rows = []
    for n in sizes:
        Xtr, ytr, _ = to_arrays(train_df.iloc[:n])
        nm, ns = compute_naive_stats(ytr)
        res: dict[str, tuple] = {}
        for name in a.models:
            t0 = time.time()
            pred = MODELS[name](Xtr, ytr, Xva)
            sc = gll(yva, pred, ids_va, nm, ns)
            res[name] = (pred, sc, bootstrap_scores(yva, pred, ids_va, nm, ns, boots), time.time() - t0)
        ref = res.get("ridge-const")
        for name, (pred, sc, bs, dt) in res.items():
            row = {"n_train": len(Xtr), "model": name, "gll": round(sc, 4),
                   "ci05": round(float(np.percentile(bs, 5)), 4), "ci95": round(float(np.percentile(bs, 95)), 4),
                   "p_better_than_ridge": round(float(np.mean(bs > ref[2])), 2) if ref and name != "ridge-const" else None,
                   "sigma_med": float(np.median(pred.sigma)), "seconds": round(dt, 1)}
            rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare.csv", index=False)
    print(f"-> {out / 'compare.csv'}")

    if a.mlflow:
        import mlflow
        mlflow.set_experiment("ariel-model-comparison")
        for name in a.models:
            with mlflow.start_run(run_name=f"compare-{name}"):
                mlflow.set_tags({"model_name": name, "git_commit": git_commit(), "val_planets": str(len(Xva))})
                for _, r in table[table.model == name].iterrows():
                    mlflow.log_metric("gll_score", r["gll"], step=int(r["n_train"]))
                    mlflow.log_metric("gll_ci05", r["ci05"], step=int(r["n_train"]))
                    mlflow.log_metric("gll_ci95", r["ci95"], step=int(r["n_train"]))
                mlflow.log_artifact(str(out / "compare.csv"))


if __name__ == "__main__":
    main()
