"""
compare_features.py — Le Ridge lui-même fait-il mal ? Variantes d'entrées et baselines simples
===============================================================================================
Le diagnostic montre que l'erreur est surtout un décalage de niveau par planète, et que pour
quelques planètes atypiques (profondeurs extrêmes) le Ridge est PIRE que l'extraction brute.
Cause suspectée : la cible wl est une profondeur (≈ Rp²) alors que le Ridge reçoit Rp (sa racine) ;
une régression linéaire sur Rp sous-estime les profondeurs extrêmes. Ce script le teste, sans
rien retélécharger, sur data/processed/incremental.

Chaque variante de μ est combinée à TROIS façons de calibrer σ (écart-type des résidus par longueur
d'onde), pour séparer l'effet de μ de l'effet de σ :
  insample : résidus d'entraînement (recette de v0.1.0) ; kfold5 : hors-échantillon sur 5 plis ;
  loo      : hors-échantillon « leave-one-out » (σ le plus proche de l'erreur du modèle final).
Variantes de μ :
  ridge, entrées Rp / Rp² ; avec ou sans features astro ; astro écrêtées (1er-99e percentile) ;
  k·Rp² (1 paramètre) ; affine par canal  a_j + b_j·Rp_j²  (2 paramètres par canal).
La ligne de référence « v0.1.0 exact » reproduit le Ridge de la boucle incrémentale (σ en échantillon).

    uv run python -m scripts.compare_features
    uv run python -m scripts.compare_features --alphas 1 10 100 --n-train 140
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from scripts.compare_models import bootstrap_scores, gll, load_spectra, to_arrays
from src.ml.features import compute_naive_stats
from src.models.hetero import Prediction, ridge_const

N_WL = 283


# --------------------------------------------------------------------------- variantes de μ

def ridge_variant(alpha: float, power: int = 1, astro: bool = True, clip: bool = False):
    """μ par Ridge sur [Rp**power (283), astro (6, optionnelles, écrêtables)]."""
    def build(X, lo=None, hi=None):
        parts = [X[:, :N_WL] ** power]
        if astro:
            a = X[:, N_WL:]
            parts.append(np.clip(a, lo, hi) if clip else a)
        return np.hstack(parts)

    def f(Xtr, ytr, Xte):
        lo, hi = (np.percentile(Xtr[:, N_WL:], [1, 99], axis=0) if clip else (None, None))
        A, B = build(Xtr, lo, hi), build(Xte, lo, hi)
        sc = StandardScaler().fit(A)
        return Ridge(alpha=alpha).fit(sc.transform(A), ytr).predict(sc.transform(B))
    return f


def scaled_identity(Xtr, ytr, Xte):
    """μ_j = k · Rp_j²   avec k = médiane de wl / Rp² sur l'entraînement (un seul paramètre)."""
    d = Xtr[:, :N_WL] ** 2
    ok = d > 1e-9
    k = float(np.median(ytr[ok] / d[ok]))
    return k * Xte[:, :N_WL] ** 2


def channel_linear(Xtr, ytr, Xte):
    """μ_j = a_j + b_j · Rp_j²  (moindres carrés par canal, vectorisé)."""
    d = Xtr[:, :N_WL] ** 2
    dm, ym = d.mean(axis=0), ytr.mean(axis=0)
    b = ((d - dm) * (ytr - ym)).sum(axis=0) / np.maximum(((d - dm) ** 2).sum(axis=0), 1e-30)
    return ym + b * (Xte[:, :N_WL] ** 2 - dm)


# --------------------------------------------------------------------------- σ hors-échantillon

def with_oof_sigma(mu_fn, Xtr, ytr, Xte, n_splits: int = 5, seed: int = 0) -> Prediction:
    """μ de mu_fn ; σ_j = écart-type des résidus hors-échantillon (K-fold) du même modèle."""
    oof = np.zeros_like(ytr)
    for tr, te in KFold(n_splits=min(n_splits, len(Xtr)), shuffle=True, random_state=seed).split(Xtr):
        oof[te] = mu_fn(Xtr[tr], ytr[tr], Xtr[te])
    sigma = np.maximum((ytr - oof).std(axis=0), 1e-6)
    mu = mu_fn(Xtr, ytr, Xte)
    return Prediction(mu, np.tile(sigma, (len(mu), 1)))


def with_sigma(mu_fn, Xtr, ytr, Xte, mode: str = "insample", seed: int = 0) -> Prediction:
    """μ de mu_fn ; σ_j calibré selon `mode` : insample | kfold5 | loo."""
    if mode == "insample":
        mu_tr = mu_fn(Xtr, ytr, Xtr)
        mu = mu_fn(Xtr, ytr, Xte)
        return Prediction(mu, np.tile(np.maximum((ytr - mu_tr).std(axis=0), 1e-6), (len(mu), 1)))
    return with_oof_sigma(mu_fn, Xtr, ytr, Xte, n_splits={"kfold5": 5, "loo": len(Xtr)}[mode], seed=seed)


SIGMA_MODES = ("insample", "kfold5", "loo")


def variants(alphas) -> dict:
    v = {}
    for a in alphas:
        v[f"ridge Rp + astro  α={a:g}"] = ridge_variant(a, 1, True)
        v[f"ridge Rp seul     α={a:g}"] = ridge_variant(a, 1, False)
        v[f"ridge Rp² + astro α={a:g}"] = ridge_variant(a, 2, True)
        v[f"ridge Rp² seul    α={a:g}"] = ridge_variant(a, 2, False)
        v[f"ridge Rp² + astro écrêtées α={a:g}"] = ridge_variant(a, 2, True, clip=True)
    v["k·Rp² (1 paramètre)"] = scaled_identity
    v["affine par canal a_j + b_j·Rp_j²"] = channel_linear
    return v


def level_stats(yva, mu, ids, watch) -> tuple[float, float]:
    """RMSE global, et |erreur de niveau| moyenne sur les planètes surveillées (les plus mal prédites)."""
    E = yva - mu
    lvl = np.abs(E.mean(axis=1))
    sel = np.isin(ids, list(watch))
    return float(np.sqrt((E ** 2).mean())), float(lvl[sel].mean()) if sel.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/incremental")
    ap.add_argument("--alphas", type=float, nargs="*", default=[10, 100])
    ap.add_argument("--sigma-modes", nargs="*", default=list(SIGMA_MODES), choices=SIGMA_MODES)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--watch", type=int, nargs="*", default=[340799832, 2199983617, 2033197078])
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/compare")
    a = ap.parse_args()

    train_df, val_df = load_spectra(Path(a.root))
    if a.n_train:
        train_df = train_df.iloc[:a.n_train]
    Xtr, ytr, _ = to_arrays(train_df)
    Xva, yva, ids = to_arrays(val_df)
    nm, ns = compute_naive_stats(ytr)
    print(f"{len(Xtr)} planètes d'entraînement, {len(Xva)} en validation")

    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, len(Xva), len(Xva)) for _ in range(a.n_boot)]

    ref = ridge_const(Xtr, ytr, Xva)                                   # v0.1.0 exact
    ref_bs = bootstrap_scores(yva, ref, ids, nm, ns, boots)
    rmse, lvl3 = level_stats(yva, ref.mu, ids, a.watch)
    def row(name, mode, pred, bs, p_better):
        rmse, lvl3 = level_stats(yva, pred.mu, ids, a.watch)
        return {"modèle": name, "σ": mode, "gll": gll(yva, pred, ids, nm, ns), "ci05": np.percentile(bs, 5),
                "ci95": np.percentile(bs, 95), "p_mieux_que_v010": p_better, "rmse": rmse,
                "niveau_3_pires": lvl3, "σ_sur_rmse": float(np.median(pred.sigma) / rmse)}

    rows = [row("v0.1.0 exact (Rp + astro, α=10)", "insample", ref, ref_bs, np.nan)]
    for name, fn in variants(a.alphas).items():
        for mode in a.sigma_modes:
            pred = with_sigma(fn, Xtr, ytr, Xva, mode)
            bs = bootstrap_scores(yva, pred, ids, nm, ns, boots)
            rows.append(row(name, mode, pred, bs, float(np.mean(bs > ref_bs))))

    table = pd.DataFrame(rows).sort_values("gll", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.float_format", lambda v: f"{v:.4g}")
    print("\n" + table.to_string(index=False))
    print(f"\nniveau_3_pires = |erreur de niveau| moyenne sur {a.watch} ; v0.1.0 est la référence de p_mieux_que_v010."
          "\nσ_sur_rmse = σ médian / rmse de validation : < 1 σ trop optimiste, > 1 σ trop prudent (idéal ≈ 1).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "features.csv", index=False)
    print(f"-> {out / 'features.csv'}")


if __name__ == "__main__":
    main()
