"""
compare_filter.py — Quel seuil pour le filtre de qualité à l'entraînement ? (jeu de test COMMUN à tous les seuils)
================================================================================================================
Le filtre `--level-range` écarte de l'entraînement les planètes dont médiane(Rp²/label) sort d'une plage. Comparer deux
seuils en évaluant chacun sur « ses propres » planètes serait trompeur : un filtre plus strict retire aussi les cas
difficiles du test. Ici :

  - mêmes plis (validation croisée répétée) pour toutes les plages ;
  - ENTRAÎNEMENT : planètes du pli d'entraînement dont le niveau est dans la plage testée ;
  - TEST : toujours les mêmes planètes du pli de test, celles dont le niveau est dans --eval-range (défaut 0,5–2) ;
  - le score est normalisé avec les mêmes statistiques naïves pour toutes les plages (planètes d'entraînement du
    pli dans --eval-range) : seuls les modèles diffèrent.
  - colonne `cv_toutes` : score sur TOUTES les planètes du pli de test, y compris celles hors --eval-range
    (le risque réel sur un jeu de test non filtré ; dominé par les échecs d'extraction extrêmes).

Même Ridge (α = 10) et σ constant que le champion.

    uv run python -m scripts.compare_filter --root $R/spectra_all
    uv run python -m scripts.compare_filter --root $R/spectra_all --ranges 0.5,2 0.6,1.7 0.7,1.5

Sortie : tableau + reports/compare/filter_cv.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_extraction import fold_splits, paired_summary
from scripts.compare_models import gll, load_spectra, to_arrays
from scripts.compare_pca import N_WL, level_of, ridge_fit_predict
from src.ml.features import compute_naive_stats
from src.models.hetero import Prediction

DEFAULT_RANGES = [(0.5, 2.0), (0.6, 1.7), (0.7, 1.5), (0.75, 1.4)]


def range_name(r: tuple[float, float] | None) -> str:
    return "sans filtre" if r is None else f"[{r[0]:g} ; {r[1]:g}]"


def in_range(lev: np.ndarray, r: tuple[float, float] | None) -> np.ndarray:
    return np.ones(len(lev), bool) if r is None else (lev >= r[0]) & (lev <= r[1])


def run_cv(X, y, ids, splits, ranges, eval_range=(0.5, 2.0)) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Scores par plage : {nom: {"commun": scores par pli, "toutes": scores par pli}} et taille moyenne d'entraînement."""
    lev = level_of(X, y)
    common = in_range(lev, eval_range)
    D = X[:, :N_WL] ** 2
    F = np.hstack([D, X[:, N_WL:]])
    out = {range_name(r): {"commun": [], "toutes": []} for r in ranges}
    sizes = {range_name(r): [] for r in ranges}
    for a, b in splits:
        a_ref = a[common[a]]
        nm, ns = compute_naive_stats(y[a_ref])
        b_common = b[common[b]]
        for r in ranges:
            tr = a[in_range(lev[a], r)]
            pred = ridge_fit_predict(F[tr], y[tr], F[b])
            keep = np.isin(b, b_common)
            sub = Prediction(pred.mu[keep], pred.sigma[keep])
            name = range_name(r)
            out[name]["commun"].append(gll(y[b_common], sub, ids[b_common], nm, ns))
            out[name]["toutes"].append(gll(y[b], pred, ids[b], nm, ns))
            sizes[name].append(len(tr))
    return ({k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()},
            {k: float(np.mean(v)) for k, v in sizes.items()})


def summarize(scores: dict[str, dict[str, np.ndarray]], sizes: dict[str, float], reference: str) -> pd.DataFrame:
    rows = []
    for name, d in scores.items():
        rows.append({"plage": name, "n_train_moy": sizes[name], "cv_commun": float(d["commun"].mean()),
                     "cv_sd": float(d["commun"].std()), **paired_summary(d["commun"] - scores[reference]["commun"]),
                     "cv_toutes": float(d["toutes"].mean())})
    return pd.DataFrame(rows).sort_values("cv_commun", ascending=False).reset_index(drop=True)


def parse_range(text: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in text.split(","))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/spectra_v2")
    ap.add_argument("--ranges", nargs="+", type=parse_range, default=DEFAULT_RANGES, metavar="MIN,MAX",
                    help="plages à comparer, ex. 0.5,2 0.7,1.5 (« sans filtre » est toujours ajouté)")
    ap.add_argument("--eval-range", type=parse_range, default=(0.5, 2.0), metavar="MIN,MAX")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/compare")
    a = ap.parse_args()
    train, val = load_spectra(Path(a.root))
    X, y, ids = to_arrays(pd.concat([val, train], ignore_index=True))
    ranges = [None] + list(a.ranges)
    lev = level_of(X, y)
    splits = fold_splits(len(X), a.folds, a.repeats, a.seed)
    print(f"{len(X)} planètes ; test commun = niveau dans {a.eval_range} ({int(in_range(lev, a.eval_range).sum())} planètes) ; "
          f"{a.folds} plis × {a.repeats} répétitions")
    for r in ranges:
        print(f"  {range_name(r):>14} : {int((~in_range(lev, r)).sum())} planètes écartées de l'entraînement")
    ref = range_name(tuple(a.eval_range))
    scores, sizes = run_cv(X, y, ids, splits, ranges, tuple(a.eval_range))
    if ref not in scores:
        ref = range_name(ranges[1])
    table = summarize(scores, sizes, ref)
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
    print(f"\nréférence : {ref}\n")
    print(table.to_string(index=False))
    print("\ncv_commun : même population de test pour toutes les plages ; diff_moy et IC 90 % : écart apparié à la référence "
          "(plis non indépendants : optimiste) ;\ncv_toutes : test incluant les planètes hors --eval-range (risque sur un test non filtré).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "filter_cv.csv", index=False)
    print(f"-> {out / 'filter_cv.csv'}")


if __name__ == "__main__":
    main()
