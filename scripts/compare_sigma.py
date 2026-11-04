"""
compare_sigma.py — σ proportionnel à la profondeur du transit : σ_ij = ρ̂(d_i) · q_j
====================================================================================
Constat : les erreurs de niveau des planètes atypiques font 16-28 % de leur profondeur, et ~15 %
en moyenne. L'erreur serait donc RELATIVE (proportionnelle à la profondeur d_i), alors que v0.1.0
utilise un σ constant : trop petit pour une planète très profonde, trop grand pour une peu profonde.

Modèle (3 paramètres + une forme par canal) :
    d_i  = médiane sur les canaux de Rp_ij²                  (observable à la prédiction)
    ρ_i  = rms des résidus de la planète i ;  ln ρ_i = a + γ·ln d_i   (régression, γ ∈ [0, 2])
    q_j  = rms sur les planètes de r_ij / ρ̂_i                (forme du bruit selon la longueur d'onde)
    σ_ij = ρ̂(d_i) · q_j
γ = 0 redonne un σ constant (contrôle) ; γ = 1 : erreur strictement relative ; « γ ajusté » : appris.

Les résidus viennent soit de l'entraînement (insample, recette de v0.1.0), soit du leave-one-out (loo).
Colonnes z_peu_profonde / z_profonde : rms de (y − μ)/σ sur le tiers de validation le moins / le plus
profond ; l'idéal est 1 partout. Un σ constant donne z < 1 pour les peu profondes et z > 1 pour
les profondes.

    uv run python -m scripts.compare_sigma
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from scripts.compare_features import N_WL, level_stats, ridge_variant
from scripts.compare_models import bootstrap_scores, gll, load_spectra, to_arrays
from src.ml.features import compute_naive_stats
from src.models.hetero import Prediction, ridge_const


def depth_of(X: np.ndarray) -> np.ndarray:
    """Profondeur observée d'une planète : médiane sur les canaux de Rp² (les 283 premières colonnes de X)."""
    return np.maximum(np.median(X[:, :N_WL] ** 2, axis=1), 1e-6)


def residuals(mu_fn, Xtr, ytr, mode: str, seed: int = 0) -> np.ndarray:
    if mode == "insample":
        return ytr - mu_fn(Xtr, ytr, Xtr)
    r = np.zeros_like(ytr)
    for tr, te in KFold(n_splits=len(Xtr), shuffle=True, random_state=seed).split(Xtr):
        r[te] = mu_fn(Xtr[tr], ytr[tr], Xtr[te])
    return ytr - r


def fit_depth_sigma(resid: np.ndarray, d: np.ndarray, gamma: float | None = None) -> dict:
    """Ajuste ρ̂(d) = exp(a)·d^γ et la forme q_j. gamma=None : γ appris (borné à [0, 2])."""
    rho = np.maximum(np.sqrt((resid ** 2).mean(axis=1)), 1e-12)
    x, y = np.log(d), np.log(rho)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    if gamma is None:
        gamma = float(np.clip(np.polyfit(x, y, 1)[0], 0.0, 2.0)) if np.std(x) > 0 else 0.0
    a = float(np.mean(y - gamma * x))
    rho_hat = np.exp(a + gamma * x)
    q = np.sqrt(((resid / rho_hat[:, None]) ** 2).mean(axis=0))
    return {"a": a, "gamma": float(gamma), "q": q, "r2": r2}


def sigma_matrix(fit: dict, d_new: np.ndarray) -> np.ndarray:
    rho_hat = np.exp(fit["a"] + fit["gamma"] * np.log(d_new))
    return np.maximum(rho_hat[:, None] * fit["q"][None, :], 1e-6)


def z_by_depth(y, pred: Prediction, d) -> tuple[float, float]:
    """rms de (y − μ)/σ sur le tiers le moins profond et le tiers le plus profond."""
    z2 = ((y - pred.mu) / pred.sigma) ** 2
    lo, hi = np.quantile(d, [1 / 3, 2 / 3])
    return float(np.sqrt(z2[d <= lo].mean())), float(np.sqrt(z2[d >= hi].mean()))


def mu_variants() -> dict:
    return {
        "Rp + astro α=10 (μ de v0.1.0)": ridge_variant(10, 1, True),
        "Rp² + astro α=10": ridge_variant(10, 2, True),
        "Rp² + astro α=100": ridge_variant(100, 2, True),
        "Rp² seul α=10": ridge_variant(10, 2, False),
    }


SIGMA_SPECS = (("const (γ=0)", 0.0), ("relatif (γ=1)", 1.0), ("γ ajusté", None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/incremental")
    ap.add_argument("--n-train", type=int, default=None)
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
    d_tr, d_va = depth_of(Xtr), depth_of(Xva)
    print(f"{len(Xtr)} planètes d'entraînement, {len(Xva)} en validation ; "
          f"profondeur d : de {d_tr.min():.2g} à {d_tr.max():.2g} (médiane {np.median(d_tr):.2g})")

    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, len(Xva), len(Xva)) for _ in range(a.n_boot)]
    ref = ridge_const(Xtr, ytr, Xva)
    ref_bs = bootstrap_scores(yva, ref, ids, nm, ns, boots)

    def row(name, sig, pred, bs, p, fit=None):
        rmse, _ = level_stats(yva, pred.mu, ids, [])
        z_lo, z_hi = z_by_depth(yva, pred, d_va)
        return {"μ": name, "σ": sig, "gll": gll(yva, pred, ids, nm, ns), "ci05": np.percentile(bs, 5),
                "ci95": np.percentile(bs, 95), "p_mieux_que_v010": p, "rmse": rmse,
                "σ_sur_rmse": float(np.median(pred.sigma) / rmse), "z_peu_profonde": z_lo, "z_profonde": z_hi,
                "γ": fit["gamma"] if fit else np.nan, "R²_ln": fit["r2"] if fit else np.nan}

    rows = [row("v0.1.0 exact", "const insample", ref, ref_bs, np.nan)]
    for name, fn in mu_variants().items():
        mu_te = fn(Xtr, ytr, Xva)
        for src in ("insample", "loo"):
            res = residuals(fn, Xtr, ytr, src)
            for label, g in SIGMA_SPECS:
                fit = fit_depth_sigma(res, d_tr, g)
                pred = Prediction(mu_te, sigma_matrix(fit, d_va))
                bs = bootstrap_scores(yva, pred, ids, nm, ns, boots)
                rows.append(row(name, f"{label} · {src}", pred, bs, float(np.mean(bs > ref_bs)), fit))

    table = pd.DataFrame(rows).sort_values("gll", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 250, "display.max_colwidth", 40, "display.float_format", lambda v: f"{v:.4g}")
    print("\n" + table.to_string(index=False))
    print("\nz_* : rms de (y−μ)/σ par tiers de profondeur de validation (idéal ≈ 1) ; γ, R²_ln : exposant et qualité "
          "de la régression ln ρ ~ ln d sur l'entraînement.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sigma_depth.csv", index=False)
    print(f"-> {out / 'sigma_depth.csv'}")


if __name__ == "__main__":
    main()
