"""Tests du σ proportionnel à la profondeur : erreur relative → γ ≈ 1 et meilleur NLL qu'un σ constant."""
import numpy as np
import scipy.stats

from scripts.compare_sigma import (N_WL, depth_of, fit_depth_sigma, residuals, sigma_matrix, z_by_depth)
from scripts.compare_features import ridge_variant
from src.models.hetero import Prediction


def make(n, seed, rel=0.15, floor=0.0):
    """Profondeurs log-normales ; erreur de niveau par planète = rel × profondeur (+ plancher constant)."""
    rng = np.random.default_rng(seed)
    d = np.exp(rng.normal(np.log(0.015), 0.6, size=(n, 1)))
    rp = np.sqrt(d) * (1 + 0.02 * rng.normal(size=(n, N_WL)))
    y = d * (1 + rel * rng.normal(size=(n, 1))) + floor * rng.normal(size=(n, N_WL)) + 1e-5 * rng.normal(size=(n, N_WL))
    return np.hstack([rp, rng.normal(size=(n, 6))]), y


def mean_ll(y, p):
    return float(scipy.stats.norm.logpdf(y, loc=p.mu, scale=p.sigma).mean())


def test_profondeur_observee():
    X, _ = make(10, 0)
    d = depth_of(X)
    assert d.shape == (10,) and (d > 0).all()


def test_gamma_proche_de_1_si_erreur_relative():
    X, y = make(150, 0, rel=0.15)
    res = residuals(ridge_variant(10, 2, False), X, y, "insample")
    fit = fit_depth_sigma(res, depth_of(X))
    # R² modeste même quand la relation est exacte : l'erreur de niveau est tirée au hasard par planète
    assert 0.6 < fit["gamma"] <= 1.4 and fit["r2"] > 0.1


def test_gamma_proche_de_0_si_erreur_constante():
    X, y = make(150, 1, rel=0.0, floor=1e-3)
    res = residuals(ridge_variant(10, 2, False), X, y, "insample")
    assert fit_depth_sigma(res, depth_of(X))["gamma"] < 0.4


def test_gamma_zero_redonne_un_sigma_constant_par_canal():
    X, y = make(100, 2)
    res = residuals(ridge_variant(10, 2, False), X, y, "insample")
    fit = fit_depth_sigma(res, depth_of(X), gamma=0.0)
    S = sigma_matrix(fit, depth_of(X))
    assert np.allclose(S, S[0]) and (S > 0).all() and S.shape == (100, N_WL)


def test_sigma_proportionnel_bat_sigma_constant_et_corrige_les_z():
    Xtr, ytr = make(150, 3)
    Xte, yte = make(120, 4)
    fn = ridge_variant(10, 2, False)
    res = residuals(fn, Xtr, ytr, "insample")
    mu = fn(Xtr, ytr, Xte)
    p0 = Prediction(mu, sigma_matrix(fit_depth_sigma(res, depth_of(Xtr), 0.0), depth_of(Xte)))
    p1 = Prediction(mu, sigma_matrix(fit_depth_sigma(res, depth_of(Xtr), None), depth_of(Xte)))
    assert mean_ll(yte, p1) > mean_ll(yte, p0) + 0.2
    lo0, hi0 = z_by_depth(yte, p0, depth_of(Xte))
    lo1, hi1 = z_by_depth(yte, p1, depth_of(Xte))
    assert hi0 > 1.3 * lo0                                   # σ constant : mal calibré selon la profondeur
    assert abs(np.log(hi1 / lo1)) < abs(np.log(hi0 / lo0))   # σ ∝ profondeur : bien mieux équilibré


def test_residuals_loo_plus_grands_que_insample_pour_un_modele_peu_regularise():
    X, y = make(40, 5)
    fn = ridge_variant(0.1, 1, True)
    r_in, r_loo = residuals(fn, X, y, "insample"), residuals(fn, X, y, "loo")
    assert np.sqrt((r_loo ** 2).mean()) > np.sqrt((r_in ** 2).mean())
