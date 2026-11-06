"""
compare_sigma_cv.py — Quelle incertitude σ pour le champion ? Comparaison appariée en validation croisée
=======================================================================================================
Le diagnostic des planètes aberrantes montre que quelques planètes (profondes ou mal extraites) tirent la
moyenne vers le bas : leur erreur dépasse largement le σ prédit (log-vraisemblance quadratique). Ce script
compare des façons de calibrer σ, SANS toucher à μ (Ridge Rp² + astro, α = 10, ramené à ≥ 0) :

  - const · insample    σ_j = écart des résidus d'entraînement (le champion v0.2.0)
  - const · loo         idem, mais résidus hors échantillon (leave-one-out) : plus proches de l'erreur réelle
  - × 1,5 / × 2         le σ du champion multiplié par un facteur (plus prudent)
  - relatif γ=1         σ proportionnel à la profondeur de la planète : σ_ij = ρ̂(d_i)·q_j
  - γ ajusté            exposant appris sur l'entraînement

Évaluation : validation croisée répétée sur TOUTES les planètes de data/processed/spectra_v2, mêmes plis pour
toutes les variantes ; différences appariées avec le champion (la validation fixe de 20 planètes est trop bruitée).

    uv run python -m scripts.compare_sigma_cv
    uv run python -m scripts.compare_sigma_cv --repeats 5

Sortie : tableau + reports/compare/sigma_cv.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_extraction import fold_splits, paired_summary
from scripts.compare_features import ridge_variant
from scripts.compare_models import gll, load_spectra, to_arrays
from scripts.compare_sigma import depth_of, fit_depth_sigma, residuals, sigma_matrix
from src.ml.features import compute_naive_stats
from src.models.hetero import Prediction

REFERENCE = "const · insample (champion)"
CONFIGS = [  # (nom, résidus, γ, facteur)
    (REFERENCE, "insample", 0.0, 1.0),
    ("const · insample × 1,5", "insample", 0.0, 1.5),
    ("const · insample × 2", "insample", 0.0, 2.0),
    ("const · loo", "loo", 0.0, 1.0),
    ("relatif γ=1 · insample", "insample", 1.0, 1.0),
    ("relatif γ=1 · loo", "loo", 1.0, 1.0),
    ("γ ajusté · insample", "insample", None, 1.0),
    ("γ ajusté · loo", "loo", None, 1.0),
]


def fold_predictions(Xa, ya, Xb, configs=CONFIGS, alpha: float = 10.0) -> dict[str, Prediction]:
    """Prédictions du pli (μ commun, σ propre à chaque variante) ; les résidus LOO ne sont calculés qu'au besoin."""
    mu_fn = ridge_variant(alpha, 2, True)
    mu = np.maximum(mu_fn(Xa, ya, Xb), 0.0)
    res = {src: residuals(mu_fn, Xa, ya, src) for src in {c[1] for c in configs}}
    d_a, d_b = depth_of(Xa), depth_of(Xb)
    return {name: Prediction(mu, k * sigma_matrix(fit_depth_sigma(res[src], d_a, gamma), d_b))
            for name, src, gamma, k in configs}


def run_cv(X, y, ids, splits, configs=CONFIGS) -> dict[str, np.ndarray]:
    scores: dict[str, list[float]] = {c[0]: [] for c in configs}
    for a, b in splits:
        nm, ns = compute_naive_stats(y[a])
        for name, pred in fold_predictions(X[a], y[a], X[b], configs).items():
            scores[name].append(gll(y[b], pred, ids[b], nm, ns))
    return {k: np.array(v) for k, v in scores.items()}


def summarize(scores: dict[str, np.ndarray], reference: str = REFERENCE) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        rows.append({"σ": name, "cv_moy": float(s.mean()), "cv_sd": float(s.std()), "cv_min": float(s.min()),
                     **paired_summary(s - scores[reference])})
    return pd.DataFrame(rows).sort_values("cv_moy", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/spectra_v2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/compare")
    a = ap.parse_args()
    train, val = load_spectra(Path(a.root))
    X, y, ids = to_arrays(pd.concat([val, train], ignore_index=True))
    splits = fold_splits(len(X), a.folds, a.repeats, a.seed)
    print(f"{len(X)} planètes, {a.folds} plis × {a.repeats} répétitions ({len(splits)} plis), μ identique pour toutes les variantes")
    table = summarize(run_cv(X, y, ids, splits))
    pd.set_option("display.width", 200, "display.max_colwidth", 40, "display.float_format", lambda v: f"{v:.4f}")
    print("\n" + table.to_string(index=False))
    print(f"\ndiff_moy = score CV de la variante − score CV du champion ({REFERENCE}), mêmes plis ; plis_gagnes = part des "
          "plis où elle gagne ;\nIC 90 % par bootstrap sur les plis (plis non indépendants : optimiste) ; cv_min = pire pli.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sigma_cv.csv", index=False)
    print(f"-> {out / 'sigma_cv.csv'}")


if __name__ == "__main__":
    main()
