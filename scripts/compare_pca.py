"""
compare_pca.py — Un sous-espace PCA des spectres cibles aide-t-il ? (idée de la solution classée 4e, Hansen 2025)
==============================================================================================================
Les spectres cibles vivent dans un sous-espace de faible dimension (quelques molécules dominent). Deux usages testés :

  1. DÉBRUITER : on projette la FORME du spectre extrait (spectre moins sa moyenne) sur les k premières composantes
     des spectres cibles d'entraînement, on garde le niveau (moyenne) tel quel :
         remplace k   Ridge sur [spectre projeté, astro]
         ajoute k     Ridge sur [spectre brut, spectre projeté, astro]
         route k      « ajoute k » si la planète est proche du sous-espace, sinon le champion (voir 2.)
  2. DÉTECTER les planètes atypiques SANS label : distance relative entre la forme extraite et sa projection,
     seuil = 90e centile des distances d'entraînement. Un spectre mal extrait ou hors du domaine d'entraînement
     en est loin. (Le filtre sur le niveau d'extraction, lui, exige les labels : il n'existe qu'à l'entraînement.)

Les composantes sont recalculées DANS chaque pli, sur les labels d'entraînement du pli seulement : pas de fuite vers
le pli de test. Validation croisée répétée, mêmes plis pour toutes les variantes, μ ramené à ≥ 0, σ constant par
longueur d'onde (résidus d'entraînement), comme le champion.

    uv run python -m scripts.compare_pca --root data/processed/spectra_v2 --level-range 0.5 2

--level-range écarte du calcul des scores (entraînement ET test) les planètes dont médiane(Rp²/label) sort de la
plage ; leurs distances PCA sont tout de même calculées et affichées (test de détection sans label).
Sorties : reports/compare/pca_cv.csv, reports/compare/pca_distances.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.compare_extraction import fold_splits, paired_summary
from scripts.compare_models import gll, load_spectra, to_arrays
from src.ml.features import compute_naive_stats
from src.models.champion import SIGMA_FLOOR
from src.models.hetero import Prediction

N_WL = 283
REFERENCE = "champion"


def level_of(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Niveau d'extraction par planète : médiane(Rp² / label). ~1 si cohérent (utilise les labels)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y > 0, X[:, :N_WL] ** 2 / y, np.nan)
    return np.nanmedian(ratio, axis=1)


def pca_basis(y_train: np.ndarray, k: int) -> np.ndarray:
    """k premières composantes de la FORME des spectres cibles (chaque spectre moins sa moyenne), (k, 283)."""
    shape = y_train - y_train.mean(axis=1, keepdims=True)
    k = min(k, min(shape.shape))
    return np.linalg.svd(shape, full_matrices=False)[2][:k]


def project(D: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectre reconstruit (niveau gardé + forme projetée) et distance relative forme↔projection (÷ niveau)."""
    level = D.mean(axis=1, keepdims=True)
    shape = D - level
    proj = (shape @ V.T) @ V
    dist = np.linalg.norm(shape - proj, axis=1) / np.maximum(np.abs(level[:, 0]), 1e-9)
    return level + proj, dist


def ridge_fit_predict(A: np.ndarray, y: np.ndarray, B: np.ndarray, alpha: float = 10.0) -> Prediction:
    sc = StandardScaler().fit(A)
    m = Ridge(alpha=alpha).fit(sc.transform(A), y)
    sigma = np.maximum((y - m.predict(sc.transform(A))).std(axis=0), SIGMA_FLOOR)
    mu = np.maximum(m.predict(sc.transform(B)), 0.0)
    return Prediction(mu, np.tile(sigma, (len(mu), 1)))


def config_names(ks) -> list[str]:
    return [REFERENCE] + [f"{kind} k={k}" for k in ks for kind in ("remplace", "ajoute", "route")]


def fold_predictions(Xa, ya, Xb, ks=(3, 5, 7), fit_mask=None) -> dict[str, Prediction]:
    """Prédictions du pli pour chaque variante ; fit_mask : planètes d'entraînement admises dans le calcul des composantes."""
    Da, Db, astro_a, astro_b = Xa[:, :N_WL] ** 2, Xb[:, :N_WL] ** 2, Xa[:, N_WL:], Xb[:, N_WL:]
    base = ridge_fit_predict(np.hstack([Da, astro_a]), ya, np.hstack([Db, astro_b]))
    out = {REFERENCE: base}
    ok = np.ones(len(ya), bool) if fit_mask is None else fit_mask
    for k in ks:
        V = pca_basis(ya[ok], k)
        Ra, da = project(Da, V)
        Rb, db = project(Db, V)
        thr = np.percentile(da, 90)
        rep = ridge_fit_predict(np.hstack([Ra, astro_a]), ya, np.hstack([Rb, astro_b]))
        add = ridge_fit_predict(np.hstack([Da, Ra, astro_a]), ya, np.hstack([Db, Rb, astro_b]))
        far = (db > thr)[:, None]
        out[f"remplace k={k}"] = rep
        out[f"ajoute k={k}"] = add
        out[f"route k={k}"] = Prediction(np.where(far, base.mu, add.mu), np.where(far, base.sigma, add.sigma))
    return out


def run_cv(X, y, ids, splits, ks=(3, 5, 7)) -> dict[str, np.ndarray]:
    scores: dict[str, list[float]] = {n: [] for n in config_names(ks)}
    for a, b in splits:
        nm, ns = compute_naive_stats(y[a])
        for name, pred in fold_predictions(X[a], y[a], X[b], ks).items():
            scores[name].append(gll(y[b], pred, ids[b], nm, ns))
    return {k: np.array(v) for k, v in scores.items()}


def ood_distances(X, y, ids, splits, k: int = 7, fit_ok: np.ndarray | None = None) -> pd.DataFrame:
    """Distance hors échantillon de chaque planète au sous-espace (composantes calculées sans elle ; moyenne sur répétitions)."""
    acc: dict[int, list[float]] = {}
    D = X[:, :N_WL] ** 2
    for a, b in splits:
        keep = a if fit_ok is None else a[fit_ok[a]]
        _, d = project(D[b], pca_basis(y[keep], k))
        for i, v in zip(b, d):
            acc.setdefault(int(i), []).append(float(v))
    idx = sorted(acc)
    return pd.DataFrame({"planet_id": ids[idx], "dist_rel": [np.mean(acc[i]) for i in idx], "niveau": level_of(X, y)[idx]})


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probabilité qu'une planète « positive » ait un score plus grand qu'une autre (Mann–Whitney)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[positive].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(scores: dict[str, np.ndarray], reference: str = REFERENCE) -> pd.DataFrame:
    rows = [{"variante": n, "cv_moy": float(s.mean()), "cv_sd": float(s.std()), "cv_min": float(s.min()),
             **paired_summary(s - scores[reference])} for n, s in scores.items()]
    return pd.DataFrame(rows).sort_values("cv_moy", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/spectra_v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--level-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diag-k", type=int, default=7)
    ap.add_argument("--out", default="reports/compare")
    a = ap.parse_args()
    train, val = load_spectra(Path(a.root))
    X, y, ids = to_arrays(pd.concat([val, train], ignore_index=True))
    lev = level_of(X, y)
    good = np.ones(len(X), bool) if a.level_range is None else (lev >= a.level_range[0]) & (lev <= a.level_range[1])
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")

    # 1. détection sans label (toutes les planètes ; composantes calculées sur les planètes « bonnes » du pli d'entraînement)
    dist = ood_distances(X, y, ids, fold_splits(len(X), a.folds, a.repeats, a.seed), a.diag_k, good)
    print(f"{len(X)} planètes ; {int((~good).sum())} hors plage de niveau {a.level_range}")
    if (~good).any():
        print(f"détection sans label : AUC de la distance PCA pour repérer les {int((~good).sum())} planètes hors plage "
              f"= {auc(dist['dist_rel'].to_numpy(), ~good):.3f}  (1 = séparation parfaite, 0,5 = hasard)")
    dist["rang"] = dist["dist_rel"].rank(ascending=False).astype(int)
    dist["hors_plage"] = ~good
    print("\nplanètes les plus éloignées du sous-espace (k=%d) :" % a.diag_k)
    print(dist.sort_values("dist_rel", ascending=False).head(12).to_string(index=False))

    # 2. scores en validation croisée appariée, sur les planètes « bonnes »
    Xg, yg, ig = X[good], y[good], ids[good]
    table = summarize(run_cv(Xg, yg, ig, fold_splits(len(Xg), a.folds, a.repeats, a.seed), tuple(a.ks)))
    print(f"\nvalidation croisée sur {len(Xg)} planètes, {a.folds} plis × {a.repeats} répétitions, même Ridge (α=10), σ constant :\n")
    print(table.to_string(index=False))
    print("\ndiff_moy = score CV − score CV du champion, mêmes plis ; IC 90 % par bootstrap sur les plis (non indépendants : optimiste).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pca_cv.csv", index=False)
    dist.to_csv(out / "pca_distances.csv", index=False)
    print(f"-> {out / 'pca_cv.csv'}, {out / 'pca_distances.csv'}")


if __name__ == "__main__":
    main()
