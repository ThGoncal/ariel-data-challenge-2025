"""
champion.py — Modèle champion v0.2.0 : Ridge sur [Rp², astro], σ constant par longueur d'onde
============================================================================================
Choisi par validation croisée sur 152 planètes (scripts/compare_extraction.py) : features en PROFONDEUR (Rp²)
plutôt qu'en Rp, extraction à fenêtre adaptée (src/signal_processing/spectra.py), σ_j = écart-type des résidus
d'entraînement. Les prédictions sont ramenées à ≥ 0 (la métrique officielle refuse les valeurs négatives).

    bundle = fit_champion(train_df, alpha=10.0, power=2)
    pred   = predict_champion(bundle, df)          # Prediction(mu, sigma), (n, 283)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ml.features import build_feature_matrix, build_target_matrix, compute_naive_stats
from src.ml.metrics import evaluate_on_validation
from src.models.hetero import Prediction

N_WL = 283
SIGMA_FLOOR = 1e-6


def feature_matrix(df: pd.DataFrame, power: int = 2) -> tuple[np.ndarray, list[str]]:
    """[Rp_1..283 ** power, astro...] ; power=2 : profondeur de transit."""
    X, cols = build_feature_matrix(df)
    X[:, :N_WL] = X[:, :N_WL] ** power
    return X, cols


SIGMA_MODES = ("const", "depth")


def depth_of(X: np.ndarray, power: int = 2) -> np.ndarray:
    """Profondeur observée d'une planète : médiane sur les canaux de Rp² (X contient Rp**power)."""
    return np.maximum(np.median(X[:, :N_WL] ** (2.0 / power), axis=1), 1e-6)


def fit_depth_sigma(resid: np.ndarray, d: np.ndarray, gamma: float | None = None) -> dict:
    """σ_ij = ρ̂(d_i)·q_j avec ρ̂(d) = exp(a)·d^γ (γ=None : appris, borné à [0, 2] ; γ=0 : σ constant)."""
    rho = np.maximum(np.sqrt((resid ** 2).mean(axis=1)), 1e-12)
    x, y = np.log(d), np.log(rho)
    if gamma is None:
        gamma = float(np.clip(np.polyfit(x, y, 1)[0], 0.0, 2.0)) if np.std(x) > 0 else 0.0
    a = float(np.mean(y - gamma * x))
    q = np.sqrt(((resid / np.exp(a + gamma * x)[:, None]) ** 2).mean(axis=0))
    return {"a": a, "gamma": float(gamma), "q": q}


def depth_sigma(fit: dict, d: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(fit["a"] + fit["gamma"] * np.log(d))[:, None] * fit["q"][None, :], SIGMA_FLOOR)


def fit_champion(train: pd.DataFrame, alpha: float = 10.0, power: int = 2, sigma: str = "const",
                 gamma: float | None = None) -> dict:
    """sigma="const" : σ_j constant (champion v0.2.0) ; sigma="depth" : σ proportionnel à la profondeur de la planète."""
    if sigma not in SIGMA_MODES:
        raise ValueError(f"sigma inconnu : {sigma!r} (attendu : {SIGMA_MODES})")
    X, cols = feature_matrix(train, power)
    y, _ = build_target_matrix(train)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    X, y = X[ok], y[ok]
    sc = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(sc.transform(X), y)
    resid = y - model.predict(sc.transform(X))
    sig = np.maximum(resid.std(axis=0), SIGMA_FLOOR)
    fit = fit_depth_sigma(resid, depth_of(X, power), gamma) if sigma == "depth" else None
    nm, ns = compute_naive_stats(y)
    return {"scaler": sc, "model": model, "sigma": sig, "sigma_mode": sigma, "sigma_fit": fit, "feat_cols": cols,
            "power": int(power), "alpha": float(alpha), "naive_mean": nm, "naive_sigma": ns, "n_train": int(ok.sum())}


def predict_champion(bundle: dict, df: pd.DataFrame) -> Prediction:
    X, _ = feature_matrix(df, bundle["power"])
    if not np.isfinite(X).all():
        raise ValueError("features non finies : une planète a des Rp ou des paramètres astro manquants")
    mu = np.maximum(bundle["model"].predict(bundle["scaler"].transform(X)), 0.0)
    fit = bundle.get("sigma_fit")                               # absent des modèles v0.2.0 : σ constant
    if fit is not None:
        return Prediction(mu, depth_sigma(fit, depth_of(X, bundle["power"])))
    return Prediction(mu, np.tile(bundle["sigma"], (len(mu), 1)))


def score_champion(bundle: dict, val: pd.DataFrame) -> float:
    """Score GLL officiel sur `val` (planètes avec labels), normalisé avec les stats naïves du bundle."""
    X, _ = feature_matrix(val, bundle["power"])
    y, _ = build_target_matrix(val)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    pred = predict_champion(bundle, val[ok])
    return float(evaluate_on_validation(y[ok], pred.mu, pred.sigma, val["planet_id"].to_numpy()[ok],
                                        bundle["naive_mean"], bundle["naive_sigma"]))
